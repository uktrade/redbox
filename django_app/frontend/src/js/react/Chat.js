// import React, { FC } from 'react';
// import { 
//   IcClassificationBanner,
//   IcFooter,
//   IcFooterLink,
//   IcNavigationButton,
//   IcNavigationItem,
//   IcSideNavigation,
//   IcDivider,
//   IcTopNavigation,
//   IcTypography,
//   SlottedSVG,
//   IcPageHeader,
//   IcCheckbox,
//   IcSectionContainer,
//   IcAlert,
//   IcButton,
//   IcTextField
// } from '@ukic/react';

// import type { IcAlignment } from '@ukic/web-components';

// import { mdiAccount, mdiChat, mdiFileDocument, mdiPencil } from '@mdi/js';

// import "@ukic/fonts/dist/fonts.css";
// import "@ukic/react/dist/core/core.css";
// import "@ukic/react/dist/core/normalize.css";


// const Chat = () => {
//   return (
//     <>
//     <body>
//       <IcClassificationBanner />
//       <IcTopNavigation
//         appTitle="Redbox"
//         status="beta"
//         version=""
//       >
//         <IcNavigationButton label="Document" slot="buttons">
//           <SlottedSVG
//             slot="icon"
//             xmlns="http://www.w3.org/2000/svg"
//             viewBox="0 0 24 24"
//             width="24"
//             height="24" 
//             path={mdiFileDocument}
//           />
//         </IcNavigationButton>
//         <IcNavigationButton label="Chat" slot="buttons">
//           <SlottedSVG
//             slot="icon"
//             xmlns="http://www.w3.org/2000/svg"
//             viewBox="0 0 24 24"
//             width="24"
//             height="24" 
//             path={mdiChat}
//           />
//         </IcNavigationButton>
//         <IcNavigationButton label="Profile" slot="buttons">
//           <SlottedSVG
//             slot="icon"
//             xmlns="http://www.w3.org/2000/svg"
//             viewBox="0 0 24 24"
//             width="24"
//             height="24" 
//             path={mdiAccount}
//           />
//         </IcNavigationButton>
//       </IcTopNavigation>
//       <IcSectionContainer style={{paddingLeft: "400px",}}>
//       <IcTypography variant="h2" style={{ paddingTop: "100px"}} aligned="center">How can Redbox help you today?</IcTypography>
//       <IcTypography variant="p" style={{paddingTop: "20px"}} aligned="center">Select an option or type any question below.</IcTypography>
//       <br></br>
//       <IcAlert
//   variant="warning"
//   message="Redbox can make mistakes. You must check for accuracy before using the output"
// />
// <br></br>
// <IcAlert
//   variant="warning"
//   message="You can use up to, and including, official sensitive documents."
// />
//     <IcButton variant="secondary" style={{paddingTop: "50px", paddingRight: "20px"}}>Draft a meeting agenda</IcButton>
//     <IcButton variant="secondary" style={{paddingTop: "50px", paddingRight: "20px"}}>Set my work objectives</IcButton>
//     <IcButton variant="secondary" style={{paddingTop: "50px",}}>Describe the role of a Parlimentary Secretary</IcButton>
//     <div style={{display: "flex"}}>
//     <IcTextField 
//   placeholder="Please enter…" 
//   helperText="Need help? View Advanced Prompt FAQ."
//   onIcChange={(ev) => console.log(ev.detail.value)}
//   style={{ paddingTop: "100px", width: "350px"}}
//   rows="6"
//   resize="true"
// />
// <IcButton variant="primary"   style={{ paddingTop: "128px"}}>Send</IcButton>
// </div>
// </IcSectionContainer>
//       <IcSideNavigation appTitle="Redbox" status="Beta" expanded='true'>
//     <IcPageHeader heading="Redbox"></IcPageHeader>
//     <IcNavigationItem slot="primary-navigation" href="#" label="New chat">
//   </IcNavigationItem>
//   <IcDivider slot="primary-navigation"></IcDivider>
//   <IcNavigationItem slot="primary-navigation" href="#" label="Recent chats">
//   </IcNavigationItem>

//   <IcDivider slot="primary-navigation"></IcDivider>
//   <IcNavigationItem slot="primary-navigation" href="#" label="Today">
//   </IcNavigationItem>
//   <IcNavigationItem slot="primary-navigation" href="#" label="New chat">
//   <SlottedSVG
//             slot="icon"
//             xmlns="http://www.w3.org/2000/svg"
//             viewBox="0 0 24 24"
//             width="24"
//             height="24" 
//             path={mdiPencil}
//             alignment="right"
//           />
//   </IcNavigationItem>
//   <IcDivider slot="primary-navigation"></IcDivider>
//   <IcNavigationItem slot="primary-navigation" href="#" label="Previous 30 days">
//   </IcNavigationItem>
//   <IcNavigationItem slot="primary-navigation" href="#" label="Report analysis">
//   <SlottedSVG
//             slot="icon"
//             xmlns="http://www.w3.org/2000/svg"
//             viewBox="0 0 24 24"
//             width="24"
//             height="24" 
//             path={mdiPencil}
//             alignment="right"
//           />
//   </IcNavigationItem>
//   <IcNavigationItem slot="primary-navigation" href="#" label="Export opportunity">
//   <SlottedSVG
//             slot="icon"
//             xmlns="http://www.w3.org/2000/svg"
//             viewBox="0 0 24 24"
//             width="24"
//             height="24" 
//             path={mdiPencil}
//             alignment="right"
//           />
//   </IcNavigationItem>
//   <IcNavigationItem slot="primary-navigation" href="#" label="Documents to use">
//   </IcNavigationItem>
//   <IcDivider slot="primary-navigation"></IcDivider>
  
// </IcSideNavigation>
//       </body>
//       </>
//   );
// };

// export default Chat;
