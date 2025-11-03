import json
import logging
import time

from locust import HttpUser, between, events, task
from websocket import WebSocketException, create_connection

# just logging stuff like we have in redbox
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("chat_test_log.txt"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class SSOLoadTestUser(HttpUser):
    wait_time = between(1, 5)
    # enter your session id and csrftoken from UI
    session_id = "your_session_id_here"
    csrftoken = "your_token_here"
    # define your question, the id of the selected file (from UI) and filename.
    # you can also submit questions to other routes; for ex: summarise the document
    question = "@tabular how many parents are aboard?"
    selected_file_id = "5c753e30-0329-4a9f-970a-6f8ddf359f62"
    filename = "survival_dataset_subsample_20250729150219.csv"

    # flake8: noqa: C901
    # flake8: noqa: PLR0912
    @task(1)
    def chat_question(self):
        ws_endpoint = "wss://dev.redbox.uktrade.digital/ws/chat/"

        # This just makes the websocket connection - locust doesnt support websockets natively so I winged it+-
        # websocket connection relies on setting the csrftoken, in addition to sessionid
        try:
            start_time = time.time()
            ws = create_connection(ws_endpoint, cookie=f"csrftoken={self.csrftoken}; sessionid={self.session_id}")
            connect_duration = (time.time() - start_time) * 1000
            self.environment.events.request.fire(
                request_type="WebSocket Chat",
                name="Connect",
                response_time=connect_duration,
                response_length=0,
                exception=None,
            )
        except WebSocketException as e:
            logger.exception("WebSocketException")
            self.environment.events.request.fire(
                request_type="WebSocket Chat", name="Connect", response_time=0, response_length=0, exception=e
            )
            return

        # Send chat question (you need to update this with your ID for your file as demoed)
        # get message format and file id by asking a question via UI and checking websocket event messages in Dev tool.
        message_data = {
            "activities": [self.filename],
            "llm": "11",
            "message": self.question,
            "selectedFiles": [self.selected_file_id],
            "sessionId": "",
        }
        try:
            start_time = time.time()
            logger.info("question sent: %s", json.dumps(message_data))
            ws.send(json.dumps(message_data))
            send_duration = (time.time() - start_time) * 1000

            self.environment.events.request.fire(
                request_type="WebSocket Chat",
                name="Send Message with File",
                response_time=send_duration,
                response_length=len(json.dumps(message_data)),
                exception=None,
            )
        except WebSocketException as e:
            logger.exception("WebSocketException")
            self.environment.events.request.fire(
                request_type="WebSocket Chat",
                name="Send Message with File",
                response_time=0,
                response_length=0,
                exception=e,
            )
            ws.close()
            return

        # this is where response is received and processed
        streamed_content = ""
        try:
            streaming_not_complete = True
            first_token_streamed = False
            start_time = time.time()
            while streaming_not_complete:
                try:
                    response = ws.recv()
                    logger.info("there is a response")
                    streaming_not_complete = True
                except WebSocketException as e:
                    logger.warning("websocket exception")
                    self.environment.events.request.fire(
                        request_type="WebSocket Chat",
                        name="Receive",
                        response_time=(time.time() - start_time) * 1000,
                        response_length=0,
                        exception=e,
                    )
                    streaming_not_complete = False
                try:
                    response_data = json.loads(response)
                    if response_data.get("type") == "session-id":
                        logger.info("received session-id (remember this one): %s", response_data.get("data"))
                        streaming_not_complete = True
                    elif response_data.get("type") == "text":
                        response_time = (time.time() - start_time) * 1000
                        streamed_content += response_data.get("data", "")
                        logger.info("a chunk of response is: %s", response_data.get("data", ""))
                        if not first_token_streamed:
                            self.environment.events.request.fire(
                                request_type="WebSocket Chat",
                                name="Receive Start Token",
                                response_time=response_time,
                                response_length=len(streamed_content),
                                exception=None,
                            )
                            first_token_streamed = True
                        streaming_not_complete = True
                    elif response_data.get("type") == "end":
                        response_time = (time.time() - start_time) * 1000
                        logger.info("response is: %s", streamed_content)
                        self.environment.events.request.fire(
                            request_type="WebSocket Chat",
                            name="Receive End Token",
                            response_time=response_time,
                            response_length=len(streamed_content),
                            exception=None,
                        )
                        streaming_not_complete = False
                        # Using nora example but if you guys are doing something different you need to update this
                        if "Agent stopped due to iteration limit or time limit" in streamed_content:
                            logger.info("Tabular agent failed to return an answer")
                        else:
                            logger.warning("Tabular agent successfully")
                    elif response_data.get("type") == "error":
                        logger.exception("error because: %s", response_data.get("data"))
                        self.environment.events.request.fire(
                            request_type="WebSocket Chat",
                            name="Receive Error",
                            response_time=(time.time() - start_time) * 1000,
                            response_length=0,
                            exception=Exception(f"error because: {response_data.get('data')}"),
                        )
                        streaming_not_complete = False
                except json.JSONDecodeError:
                    logger.warning("failed parsing it as json because: %s", response)
                    continue
        except WebSocketException as e:
            logger.exception("exception")
            self.environment.events.request.fire(
                request_type="WebSocket Chat", name="Receive", response_time=0, response_length=0, exception=e
            )
            streaming_not_complete = False
        finally:
            ws.close()
            logger.info("websocket connection closed")


@events.test_start.add_listener
def on_test_start():
    logger.info("starting")


@events.test_stop.add_listener
def on_test_stop(environment):
    logger.info("this is finished now and failed by %s", environment.runner.stats.total.fail_ratio)
