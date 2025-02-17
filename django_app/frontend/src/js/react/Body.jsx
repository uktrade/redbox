import React from "react";
import ReactDOM from "react-dom/client";
import {
  IcClassificationBanner,
  IcNavigationButton,
  IcSectionContainer,
  IcTopNavigation,
  SlottedSVG,
  IcHero,
  IcButton,
  IcDivider,
  IcAccordion,
  IcTypography,
} from '@ukic/react';

import { mdiAccount } from '@mdi/js';

import { IcCardHorizontal } from '@ukic/canary-react';

import "@ukic/fonts/dist/fonts.css";
import "@ukic/react/dist/core/core.css";
import "@ukic/react/dist/core/normalize.css";

const Homepage = ({ security, isAuthenticated }) => {
  return (
    <>
      <IcClassificationBanner />

      <IcTopNavigation appTitle="Redbox" status="beta" version="">
        <IcNavigationButton label="Profile" slot="buttons">
          <SlottedSVG
            slot="icon"
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 24 24"
            width="24"
            height="24"
            path={mdiAccount}
          />
        </IcNavigationButton>
      </IcTopNavigation>

      <IcHero
        heading={`Use Artificial Intelligence (AI) to interact with ${security} level documents from your own document set`}
        subheading={`You may summarise documents, ask questions related to one or more documents, and chat directly with the AI to assist you in your work. You can use up to, and including, ${security} documents.`}
      >
        {!isAuthenticated && (
          <IcButton variant="primary" slot="interaction">
            Sign in
          </IcButton>
        )}
      </IcHero>

      <br />

      <IcSectionContainer gap="large" aligned="center">
        <IcCardHorizontal
          heading="Upload"
          message="Easily upload a wide range of document formats into your personal Redbox."
        />
        <IcCardHorizontal
          heading="Chat"
          message="You can communicate with your Redbox and ask questions of the documents contained within. Use Redbox to unlock deeper insights in the various documents you have uploaded, pulling together summaries of individual or combined documents."
        />
        <IcCardHorizontal
          heading="Secure"
          message={`Redbox is built in our secure and private cloud which enables you to upload, up to, and including, ${security} documents in your Redbox.`}
        />
      </IcSectionContainer>

      <IcDivider slot="primary-navigation"></IcDivider>

      <IcSectionContainer>
        <h1>Frequently Asked Questions</h1>
        <p>Responses to these FAQs about Redbox were generated using Redbox.</p>
        <IcAccordion heading="What is Redbox?">
            <IcTypography variant="body">
              Redbox is a new kind of tool for civil servants which uses Large Language Models (LLMs) to process text-based information.
              Redbox is currently only available to a small number of DBT staff as the department assesses its usefulness.
              What makes Redbox particularly useful is that it can look at any documents you upload and work with you on them.
            </IcTypography>
            </IcAccordion>
          {/* <IcAccordionItem title="What is Generative AI?">
            <p>
              Generative AI is a fairly new kind of algorithmic machine learning based on 'neural architecture'. This architecture allows
              machine learning models (called large language models in this case) to be trained on large amounts of information, and then
              be able to converse with a user by 'generating' answers to questions or assisting with text-based processing tasks.
            </p>
          </IcAccordionItem>
          <IcAccordionItem title="Do I need technical skills to use Redbox?">
            <p>
              Not at all, the interface for Redbox comprises a document upload feature and a chat feature. No technical skills are required,
              but an understanding of what large language models are and what they can do makes an enormous difference to users when interacting
              with them. The principal skill to use Redbox well comes down to a skill called 'prompt engineering', which is just natural language
              you type in and get the tool to respond. There are some great Gov.uk short courses on generative AI you may consider taking:
              <a href="https://cddo.blog.gov.uk/2024/01/19/artificial-intelligence-introducing-our-series-of-online-courses-on-generative-ai/" className="govuk-link">See the courses here</a>.
            </p>
          </IcAccordionItem>
          <IcAccordionItem title="What can I and can’t I share with Redbox?">
            <p>
              For the moment, you can share text and documents up to the {security} classification level. Take care to check before sharing
              text or documents with Redbox that they do not exceed this designation. 
              Alongside this, do not share Personal Data with Redbox. For more information on what constitutes personal data, watch this short
              <a href="https://dbis.sharepoint.com/sites/DataProtectionOSS/SitePages/Your-guide-to-understanding-personal-data.aspx" className="govuk-link">video</a>. 
              If you are unsure whether you are processing personal data, you should contact the <a href="https://workspace.trade.gov.uk/teams/data-protection-and-gdpr/" className="govuk-link">Data Protection Team</a>
               to check.
            </p>
          </IcAccordionItem>
          <IcAccordionItem title="What is a ‘hallucination’?">
            <p>
              In the context of LLMs, a hallucination refers to the generation of information that is incorrect, nonsensical, or entirely fabricated, despite
              being presented in a credible manner. This can happen when the model produces details, facts, or asserts that are not grounded in reality or the
              data it was trained on, potentially leading to misinformation or misunderstanding.
            </p>
          </IcAccordionItem>
        </IcAccordion> */}
      </IcSectionContainer>
    </>
  );
};

const rootElement = document.getElementById("root");
if (rootElement) {
    const root = ReactDOM.createRoot(rootElement);
    root.render(<Homepage security="Your security level here" isAuthenticated={false} />);
}
