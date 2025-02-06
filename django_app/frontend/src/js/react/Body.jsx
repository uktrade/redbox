import React, { FC } from 'react';
import ReactDOM from "react-dom/client";
import { 
  IcClassificationBanner,
  IcNavigationButton,
  IcSectionContainer,
  IcTopNavigation,
  SlottedSVG,
  IcHero,
  IcButton,
  IcDivider
} from '@ukic/react';

import { IcCardHorizontal } from '@ukic/canary-react';

import { mdiAccount } from '@mdi/js';
  
import "@ukic/fonts/dist/fonts.css";
import "@ukic/react/dist/core/core.css";
import "@ukic/react/dist/core/normalize.css";

const Body = () => {
  return (
    <>
      <IcClassificationBanner />
      <IcTopNavigation
        appTitle="Redbox"
        status="beta"
        version=""
      >
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
        heading="Get AI-powered insights from your Official documents"
        subheading="Ask questions, create summaries, or chat with AI about single or multiple documents to support your work."
      >
        <IcButton variant="primary" slot="interaction">
          Sign in
        </IcButton>
      </IcHero>

      <br />
      <IcSectionContainer
        style={{ display: 'flex' }}
        gap="large"
        aligned="center"
      >
        <IcCardHorizontal
          heading="Upload"
          message="Upload all common document types to your Redbox - it's that simple"
          style={{ margin: '10px' }}
        />
        <IcCardHorizontal
          heading="Chat"
          message="Ask questions about your documents and uncover insights through Redbox's AI. Get instant summaries of individual documents or combine multiple documents to discover connections across your content."
          style={{ margin: '10px' }}
        />
        <IcCardHorizontal
          heading="Secure"
          message="Redbox is built in our secure and private cloud, which enables you to upload documents up to and including Official classification in your Redbox."
          style={{ margin: '10px' }}
        />
        <IcCardHorizontal
          heading="new card"
          message="Redbox is built in our secure and private cloud, which enables you to upload documents up to and including Official classification in your Redbox."
          style={{ margin: '10px' }}
        />
      </IcSectionContainer>

      <IcSectionContainer>
        <div className="govuk-grid-row">
          <div className="govuk-grid-column-two-thirds" style={{ borderBottom: "1px solid black" }}>
            <h1 className="govuk-heading-m govuk-!-margin-top-5 govuk-!-padding-top-5">Frequently Asked Questions</h1>
          </div>
          <IcDivider slot="primary-navigation"></IcDivider>
        </div>
        <div className="govuk-accordion" data-module="govuk-accordion" id="faq">
          {/* Accordion content here */}
        </div>
      </IcSectionContainer>
    </>
  );
};

const rootElement = document.getElementById("root");
if (rootElement) {
    const root = ReactDOM.createRoot(rootElement);
    root.render(<Body />);
}