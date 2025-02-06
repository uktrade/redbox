import React from "react";
import ReactDOM from "react-dom/client";
import { 
  IcNavigationButton,
  IcTopNavigation,
  SlottedSVG,
} from '@ukic/react';

import { mdiAccount } from '@mdi/js';

import "@ukic/fonts/dist/fonts.css";
import "@ukic/react/dist/core/core.css";
import "@ukic/react/dist/core/normalize.css";

const TopNav = ({ product_name, menu_items, user_items, phase, home_path = "/" }) => {
  return (
    <>
      <IcTopNavigation appTitle={product_name} status={phase ? phase : 'alpha'} version="">
        <div className="iai-top-nav">
          <div className="iai-top-nav__container govuk-width-container">
            <div className="iai-top-nav__top-container">
              <div className="iai-top-nav__product">
                <SlottedSVG
                  slot="icon"
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 32 30"
                  width="32"
                  height="30"
                  path="M22.6 10.4c-1 .4-2-.1-2.4-1-.4-.9.1-2 1-2.4.9-.4 2 .1 2.4 1s-.1 2-1 2.4m-5.9 6.7c-.9.4-2-.1-2.4-1-.4-.9.1-2 1-2.4.9-.4 2 .1 2.4 1s-.1 2-1 2.4..." 
                />
                <div className="iai-top-nav__product-name">
                  <a className="iai-top-nav__product-link" href={home_path}>{product_name}</a>
                </div>
                {phase && (
                  <div className="iai-top-nav__phase">{phase}</div>
                )}
              </div>
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
            </div>
            <div className="mobile-drop-down">
              <nav aria-label="Menu" className="iai-top-nav__nav govuk-width-container">
                <ul id="navigation" className="iai-top-nav__nav-list">
                  {menu_items.map((menu_item, index) => (
                    <li 
                      key={index} 
                      className={`iai-top-nav__link-item ${menu_item.active ? 'iai-top-nav__link-item--active' : ''}`}
                    >
                      <a 
                        className="iai-top-nav__link" 
                        href={menu_item.href} 
                        aria-current={menu_item.active ? "page" : undefined}
                      >
                        {menu_item.text}
                      </a>
                    </li>
                  ))}
                </ul>
                {user_items && (
                  <div className="iai-top-nav__user">
                    <button className="iai-top-nav__link iai-top-nav__link--user" aria-expanded="false">
                      {user_items.initials}
                      <span className="govuk-visually-hidden">user</span>
                    </button>
                    <ul className="iai-top-nav__user-drop-down">
                      {user_items.menu_items.map((menu_item, index) => (
                        <li key={index} className="iai-top-nav__link-item iai-top-nav__user-link-item">
                          <a
                            className="iai-top-nav__link iai-top-nav__user-link"
                            href={menu_item.href}
                            aria-current={menu_item.active ? "page" : undefined}
                          >
                            {menu_item.text}
                          </a>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                <li className="iai-top-nav__link-item">
                  <a className="iai-top-nav__link" href="https://teams.microsoft.com/l/channel/19%3A9ae6b3b539724595a3139c2b16dc56ef%40thread.tacv2/Redbox%20trial%20participants%20Chat%20Channel?groupId=7a71ce78-fe77-4185-825c-ae40cb07d614&tenantId=8fa217ec-33aa-46fb-ad96-dfe68006bb86">
                    Teams Chat
                  </a>
                </li>
              </nav>
            </div>
          </div>
        </div>
      </IcTopNavigation>
    </>
  );
}

const topNavElement = document.getElementById("topNav");
if (topNavElement) {
  const productName = topNavElement.getAttribute("data-product_name");
  const menuItems = JSON.parse(topNavElement.getAttribute("data-menu_items"));
  const userItems = JSON.parse(topNavElement.getAttribute("data-user_items"));
  const phase = topNavElement.getAttribute("data-phase");
  const homePath = topNavElement.getAttribute("data-home_path");

  // Render the TopNav component with the data
  const topNav = ReactDOM.createRoot(topNavElement);
  topNav.render(
    <TopNav 
      product_name={productName}
      menu_items={menuItems}
      user_items={userItems}
      phase={phase}
      home_path={homePath}
    />
  );
}