import inspect
import os
import re
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from bertopic import BERTopic
from django.db.models import F, Prefetch
from wordcloud import STOPWORDS, WordCloud


class ChatHistoryAnalysis:
    def __init__(self) -> None:
        root = Path(__file__).parents[2]
        self.evaluation_dir = root / "notebooks/evaluation"
        results_dir = f"{self.evaluation_dir}/results"
        self.visualisation_dir = f"{results_dir}/visualisations/"
        self.table_dir = f"{results_dir}/table/"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(self.visualisation_dir, exist_ok=True)
        os.makedirs(self.table_dir, exist_ok=True)

        self.chat_logs = pd.read_csv("notebooks/evaluation/data/chat_histories/2024_09_06_chathistory.csv")

        # This column dictionary will change keys depending on names given in data frame structure
        COLUMNS_DICT = {
            "history_modified_at": "chat_created_at",
            "history_id": "message_id",
            "history_users": "user_email",
            "message_role": "role",
            "message_text": "text",
            "message_route": "route",
            "message_created_at": "prompt_created_at",
            "rating_rating": "rating",
            "rating_text": "rating_text",
        }

        # Specifying team members to remove from data
        TEAM_EMAILS = [
            "andrea.bocci",
            "natasha.boyse",
            "isobel.daley",
            "alfie.dennen",
            "anthoni.gleeson",
            "jason.kitcat",
            "will.langdale",
            "euan.soutter",
            "simon.strong",
            "sian.thomas",
        ]

        # Note I tried the following with .rename() and it did not work.
        self.chat_logs = self.chat_logs[list(COLUMNS_DICT.keys())]
        self.chat_logs.columns = [COLUMNS_DICT[col] if col in COLUMNS_DICT else col for col in self.chat_logs.columns]        
        # Select specific columns and converting to readable timestamp
        self.chat_logs = self.chat_logs[~self.chat_logs['user_email'].str.contains('|'.join(TEAM_EMAILS), na=False)].reset_index(drop=True)

        self.participant_codes = pd.read_csv("notebooks/evaluation/data/user_participants/redbox_participant_codes.csv")

        def map_rpb_number(user_email, participant_df):
            """
            Map participant codes
            """
            match = participant_df[participant_df['users'].apply(lambda x: x in user_email)]
            if not match.empty:
                return match['rpb_number'].values[0]
            else:
                return None

        self.chat_logs['user_email'] = self.chat_logs['user_email'].apply(lambda x: map_rpb_number(x, self.participant_codes))

        self.chat_logs["text"] = self.chat_logs["text"].astype(str)

        self.chat_logs["chat_created_at"] = pd.to_datetime(self.chat_logs["chat_created_at"])
        self.chat_logs["prompt_created_at"] = pd.to_datetime(self.chat_logs["prompt_created_at"])



        def backfill_route_column(df: pd.DataFrame) -> pd.DataFrame:
            """
            Function to selectively fill NaNs in 'Route' where 'Role' is 'user' using the following row with 'ai'
            """
            for i in range(len(df) - 1):
                # Check if current row has 'role' as 'user' and 'route' is NaN
                if pd.isna(df.loc[i, "route"]) and df.loc[i, "role"] == "user":
                    # Check if the next row has 'Role' as 'ai'
                    if df.loc[i + 1, "role"] == "ai":
                        # Backfill the current NaN 'route' with the 'route' from the next row
                        df.loc[i, "route"] = df.loc[i + 1, "route"]
            return df

        self.chat_logs = backfill_route_column(
            self.chat_logs.sort_values(by=["user_email", "message_id", "chat_created_at"]).reset_index(drop=True)
        )

        self.ai_responses = self.chat_logs[self.chat_logs["role"] == "ai"]
        self.user_responses = self.chat_logs[self.chat_logs["role"] == "user"]
        self.user_responses["route"] = self.user_responses["route"].fillna("none")

        self.chat_logs["tokens"] = self.chat_logs["text"].apply(self.preprocess_text)
        self.ai_responses["tokens"] = self.ai_responses["text"].apply(self.preprocess_text)
        self.user_responses["tokens"] = self.user_responses["text"].apply(self.preprocess_text)

        self.topic_model = None
        self.topic_model_over_time = None

        self.figsize = (10, 5)
        mpl.rcParams.update({"font.size": 10})
        mpl.rcParams.update({"font.family": "Arial"})

    def anonymise_users(self):
        """
        If toggle switched in streamlit app then users are anonymised.
        """
        chat_log_users = self.chat_logs["user_email"].unique()
        n_users = len(chat_log_users)
        anon_users = []
        for i in range(1, n_users + 1):
            anon_users.append(f"User {i}")
        anon_user_dict = dict(zip(chat_log_users, anon_users, strict=False))
        self.chat_logs["user_email"] = self.chat_logs["user_email"].map(anon_user_dict)
        self.ai_responses["user_email"] = self.ai_responses["user_email"].map(anon_user_dict)
        self.user_responses["user_email"] = self.user_responses["user_email"].map(anon_user_dict)

    def preprocess_text(self, text):
        tokens = text.split()
        tokens = [word.lower() for word in tokens if word.isalpha()]
        return tokens

    def get_user_frequency(self) -> pd.DataFrame:
        """
        Creates a data frame of total inputs by user.
        """
        user_counts = self.user_responses["user_email"].value_counts().reset_index(name="values")
        return user_counts

    def plot_user_frequency(self):
        """
        Generates a bar plot of total inputs per user.
        """
        user_counts = self.get_user_frequency()
        plt.figure(figsize=self.figsize)
        sns.barplot(data=user_counts, x="user_email", y="values", palette="viridis")
        plt.title("Unique Users by Total Number of Messages")
        xlabels = user_counts.user_email
        xlabels_new = ["\n".join(textwrap.wrap(name, width=10)) for name in xlabels]
        plt.xticks(range(len(xlabels_new)), xlabels_new)
        plt.xlabel("Users")
        plt.ylabel("Total No. of Prompts")

    def get_redbox_traffic(self) -> pd.DataFrame:
        """
        Returns a dataframe of redbox usage by time.
        """
        redbox_traffic_df = (
            self.user_responses["prompt_created_at"]
            .groupby(by=self.user_responses["prompt_created_at"].dt.date)
            .count()
        )

        return redbox_traffic_df

    def plot_redbox_traffic(self):
        """
        Generates a line plot of redbox usage over time
        """
        redbox_traffic_df = self.get_redbox_traffic()
        plt.figure(figsize=self.figsize)
        redbox_traffic_df.plot(kind="line")
        plt.title("Usage of Redbox AI Over Time")
        plt.xlabel("Date")
        plt.ylabel("Number of Prompts")

    def get_redbox_traffic_by_user(self, dt_grouping: str) -> pd.DataFrame:
        """
        Returns a dataframe of redbox usage by user over time.
        """
        user_responses = self.user_responses
        if dt_grouping == "week":
            redbox_traffic_by_user_df = (
                user_responses.groupby([user_responses["prompt_created_at"].dt.strftime("%W %Y"), "user_email"])
                .size()
                .unstack(fill_value=0)
            )
        else:
            redbox_traffic_by_user_df = (
                user_responses.groupby([user_responses["prompt_created_at"].dt.date, "user_email"])
                .size()
                .unstack(fill_value=0)
            )
        return redbox_traffic_by_user_df

    def plot_redbox_traffic_by_user(self):
        """
        Generates a plot of redbox usage by user over time
        """
        redbox_traffic_by_user_df = self.get_redbox_traffic_by_user(dt_grouping="day")
        plt.figure(figsize=self.figsize)
        fig = sns.lineplot(data=redbox_traffic_by_user_df, markers=True)
        fig.set_xlabel("Date")
        fig.set_ylabel("No. of Prompts")
        fig.set_title("Usage of Redbox by User over Time")
        sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1), title="Users")

    def plot_redbox_traffic_by_user_weekly(self):
        """
        Generates a plot of redbox usage by user over time
        """
        redbox_traffic_by_user_df = self.get_redbox_traffic_by_user(dt_grouping="week").reset_index()
        plt.figure(figsize=self.figsize)
        fig = sns.lineplot(data=redbox_traffic_by_user_df, markers=True)
        fig.set_xlabel("Week")
        fig.set(xticks=redbox_traffic_by_user_df.index.values)

        fig.set_ylabel("No. of Prompts")
        fig.set_title("Usage of Redbox by User over Time")
        sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1), title="Users")

    def get_user_word_frequency(self) -> pd.DataFrame:
        """
        Returns a dataframe with word frequency, removing stopwords.
        """
        all_tokens = [token for tokens in self.user_responses["tokens"] for token in tokens]
        stopwords_removed_from_all_tokens = [word for word in all_tokens if word not in STOPWORDS]
        word_freq = Counter(stopwords_removed_from_all_tokens)
        return word_freq

    def get_ai_word_frequency(self) -> pd.DataFrame:
        """
        Returns a dataframe with word frequency, removing stopwords.
        """
        ai_word_freq = Counter(
            [token for tokens in self.ai_responses["tokens"] for token in tokens if token not in STOPWORDS]
        )
        return ai_word_freq

    def plot_user_wordcloud(self):
        """
        Creates wordcloud of user prompts.
        """
        word_freq = self.get_user_word_frequency()
        # TODO: assess value
        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        plt.figure(figsize=self.figsize)
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Frequent User Words")

    def plot_top_user_word_frequency(self):
        """
        Creates bar plot of most common user words.
        """
        word_freq = self.get_user_word_frequency()
        most_common_words = word_freq.most_common(
            20  # TODO: determine how many common words we want and the right vis. for this
        )
        words, counts = zip(*most_common_words, strict=False)
        plt.figure(figsize=self.figsize)
        sns.barplot(x=list(counts), y=list(words), palette="viridis")
        plt.title("Top 20 Most Frequent User Words")
        plt.xlabel("Frequency")
        plt.ylabel("Words")

    def plot_ai_wordcloud(self):
        """
        Creates wordcloud of AI outputs.
        """
        ai_word_freq = self.get_ai_word_frequency()
        # TODO: Assess value
        ai_wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(
            ai_word_freq
        )
        plt.figure(figsize=self.figsize)
        plt.imshow(ai_wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("Most Frequent Words in AI Responses")

    def plot_top_ai_word_frequency(self):
        """
        Creates bar plot of most common user words.
        """
        ai_word_freq = self.get_ai_word_frequency()

        most_common_words = ai_word_freq.most_common(
            20
        )  # TODO: determine how many common words we want and the right vis. for this

        words, counts = zip(*most_common_words, strict=False)
        plt.figure(figsize=self.figsize)
        sns.barplot(x=list(counts), y=list(words), palette="viridis")
        plt.title("Top 20 Most Frequent AI Reponse Words")
        plt.xlabel("Frequency")
        plt.ylabel("Words")

    def get_ai_response_pattern(self):
        """
        Returns a dataframe of ai repsonse patterns.
        """

        def clean_text(
            text,
        ):
            # was including asterisks giving useless info to the graph I'm still not
            # entirely convinced on the benefit of this analysis
            return re.sub("[!@#$*]", "", text).strip()

        self.ai_responses["clean_text"] = self.ai_responses["text"].apply(clean_text)
        ai_response_patterns_df = (
            self.ai_responses["clean_text"].apply(lambda x: " ".join(x.split()[:2])).value_counts().head(10)
        )
        return ai_response_patterns_df

    def plot_ai_response_pattern(self):
        """
        Bar plot of most common ai response patterns.
        """
        ai_response_pattern_df = self.get_ai_response_pattern()
        plt.figure(figsize=self.figsize)
        sns.barplot(x=ai_response_pattern_df.values, y=ai_response_pattern_df.index, palette="magma")
        plt.title("Common Patterns in AI Responses")
        plt.xlabel("Frequency")
        plt.ylabel("Patterns")

    def get_routes_over_time(self, dt_grouping: str) -> pd.DataFrame:
        """
        Returns a dataframe of redbox usage by user over time.
        """
        user_responses = self.user_responses
        if dt_grouping == "week":
            routes_over_time = (
                user_responses.groupby([user_responses["prompt_created_at"].dt.strftime("%W %Y"), "route"])
                .size()
                .unstack(fill_value=0)
            )
        else:
            routes_over_time = (
                user_responses.groupby([user_responses["prompt_created_at"].dt.date, "route"])
                .size()
                .unstack(fill_value=0)
            )
        return routes_over_time

    def plot_routes_over_time(self):
        routes_over_time_df = self.get_routes_over_time(dt_grouping="day")
        plt.figure(figsize=self.figsize)
        fig = sns.lineplot(data=routes_over_time_df, markers=True)
        fig.set_xlabel("Date")
        fig.set_ylabel("No. of Prompts")
        fig.set_title("Popularity of Routes over Time")
        sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1), title="route")

    def plot_routes_over_time_weekly(self):
        """
        Generates a plot of redbox usage by user over time
        """
        get_routes_over_time = self.get_routes_over_time(dt_grouping="week").reset_index()
        plt.figure(figsize=self.figsize)
        fig = sns.lineplot(data=get_routes_over_time, markers=True)
        fig.set_xlabel("Week")
        fig.set(xticks=get_routes_over_time.index.values)

        fig.set_ylabel("No. of Prompts")
        fig.set_title("Popularity of Routes over Time")
        sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1), title="route")

    def get_routes(self) -> pd.DataFrame:
        """
        Returns dataframe of common user routes.
        """
        # TODO: Some users repeat the same message several times (drop?)
        # df = df.drop_duplicates(['text'])
        user_responses = self.user_responses
        user_routes_df = user_responses.groupby(["user_email"])["route"].value_counts().unstack()
        return user_routes_df

    def plot_user_routes(self):
        """
        Creates bar plot of user routes taken.
        """
        user_routes_df = self.get_routes()
        plt.figure(figsize=self.figsize)
        user_routes_df.plot(kind="bar", figsize=self.figsize)
        plt.xticks(rotation=0)
        plt.xlabel("Users")
        xlabels = user_routes_df.index
        xlabels_new = ["\n".join(textwrap.wrap(name, width=10)) for name in xlabels]
        plt.xticks(range(len(xlabels_new)), xlabels_new)
        plt.ylabel("Number of routes taken")
        plt.title("Routes taken per user")

    def get_route_transitions(self) -> pd.DataFrame:
        """
        Returns a dataframe of route transitions.
        """

        # TODO: Check this works with the groupby ID and time order of events
        def route_transitions(df):
            df["next_route"] = df["route"].shift(1)
            df["transition"] = df.apply(
                lambda row: f"{row['route']} to {row['next_route']}" if pd.notna(row["next_route"]) else None, axis=1
            )
            return df

        # Get route transitions and counts per user session (is 'id' appropriate?)
        df_transitions = self.user_responses.groupby("message_id").apply(route_transitions).reset_index(drop=True)

        route_transitions_df = df_transitions["transition"].value_counts().reset_index()
        return route_transitions_df

    def plot_route_transitions(self):
        """
        Bar plot of common route transition.
        """
        route_transitions_df = self.get_route_transitions()
        sns.barplot(x=route_transitions_df["transition"], y=route_transitions_df["count"], palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.title("Number of different route transitions")
        plt.xlabel("Route transition")
        plt.ylabel("Number of route transitions")

    def get_ratings(self) -> pd.DataFrame:
        ratings_df = self.ai_responses[self.ai_responses["rating"].notnull()]

        return ratings_df

    def rating_stats(self):
        percentage_of_ratings = round((len(self.get_ratings()) / len(self.ai_responses)) * 100, 2)
        return f"Only {percentage_of_ratings}% of AI responses have been rated"

    def plot_ratings(self):
        sns.countplot(x="rating", data=self.get_ratings(), palette="coolwarm")
        plt.title("Total Counts of AI Response Ratings", fontsize=15)
        plt.xlabel("AI Response Rating")
        plt.ylabel("Count")

    def get_topics(self):
        """
        Aims to answer: Are users asking about common topics?
        Removes stopwords and fits a simple topic model.
        """
        STOPWORDS.add("@chat")
        STOPWORDS.add("@summarise")

        text_without_stopwords = self.user_responses["text"].apply(
            lambda row: " ".join([word for word in row.split() if word not in (STOPWORDS)])
        )
        created_at = self.user_responses["chat_created_at"].to_list()

        topic_model = BERTopic(verbose=True)
        topic_model.fit_transform(text_without_stopwords)
        topics_over_time = topic_model.topics_over_time(text_without_stopwords, created_at)

        self.topic_model = topic_model
        self.topics_over_time = topics_over_time

    def plot_topics(self):
        return self.topic_model.visualize_topics(width=800, height=500)

    def plot_hierarchy(self):
        return self.topic_model.visualize_hierarchy(width=800, height=400)

    def plot_barchart(self):
        return self.topic_model.visualize_barchart(width=800, height=400)

    def plot_topics_over_time(self):
        return self.topic_model.visualize_topics_over_time(
            self.topics_over_time, top_n_topics=5, normalize_frequency=True
        )

    def get_prompt_lengths(self) -> pd.DataFrame:
        """
        Adds the prompt lengths to the user prompt column
        """
        user_responses_df = self.user_responses
        user_responses_df["no_input_words"] = user_responses_df["text"].apply(lambda n: len(n.split()))
        return user_responses_df

    def filter_prompt_lengths(self, outlier_max: int):
        """
        Creates option to filter for outliers
        """
        user_responses_df = self.get_prompt_lengths()
        user_responses_df = user_responses_df[user_responses_df["no_input_words"] < outlier_max]
        return user_responses_df

    def plot_prompt_lengths(self, outlier_max: int):
        """
        How does prompt length vary?
        """
        user_responses_df = self.filter_prompt_lengths(outlier_max)
        fig = sns.displot(user_responses_df["no_input_words"])
        fig.set_axis_labels(x_var="No. of words in prompt", y_var="Count")

    def get_prompt_length_vs_chat_length(self, outlier_max: int) -> pd.DataFrame:
        """
        Returns a dataframe comparing average prompt length with chat length.
        """
        user_responses_df = self.filter_prompt_lengths(outlier_max)
        mean_inputs_df = (
            user_responses_df[["message_id", "user_email", "no_input_words"]]
            .groupby(by=["message_id", "user_email"])
            .agg({"no_input_words": "mean"})
            .rename(columns={"no_input_words": "mean_input_words"})
            .reset_index()
        )
        no_inputs_df = (
            user_responses_df[["message_id", "user_email"]]
            .groupby("message_id")
            .value_counts()
            .reset_index(name="no_inputs")
        )
        compare_inputs_words_df = no_inputs_df.merge(
            mean_inputs_df, left_on=["message_id", "user_email"], right_on=["message_id", "user_email"]
        )

        return compare_inputs_words_df

    def plot_prompt_length_vs_chat_legnth(self, outlier_max: int):
        """
        Creates a scatterplot of prompt length vs chat legnth.
        """
        compare_inputs_words_df = self.get_prompt_length_vs_chat_length(outlier_max=outlier_max)
        fig = sns.scatterplot(data=compare_inputs_words_df, x="no_inputs", y="mean_input_words", hue="user_email")
        fig.set_xlabel("No. of prompts")
        fig.set_ylabel("Mean length of prompt")
        fig.set_title("Scatter plot comparing number of prompts with the length of prompt for each user session")
        sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1), title="Users")


def main():
    chat_history_analysis = ChatHistoryAnalysis()
    attrs = (getattr(chat_history_analysis, name) for name in dir(chat_history_analysis))
    methods = filter(inspect.ismethod, attrs)
    for method in methods:
        try:
            method()
        except TypeError:
            pass


if __name__ == "__main__":
    main()
