// @ts-check

import DOMPurify from "dompurify";

const ALLOWED_CUSTOM_TAGS = [
  "streamed-content",
  "chat-message",
  "ids-loading-message",
  "feedback-buttons",
  "copy-text",
  "ids-popover",
];

// reduced set, think twice before including hx-* attrs that allow arbitary js execution such as hx-on
const HTMX_ATTRS = ["hx-get", "hx-trigger", "hx-target", "hx-swap"];


const CUSTOM_ELEMENT_HANDLING = {
  tagNameCheck: (/** @type {String} */ tagName) =>
    ALLOWED_CUSTOM_TAGS.includes(tagName),

  attributeNameCheck: (/** @type {String} */ attr) => true,
  allowCustomizedBuiltInElements: false,
};

DOMPurify.setConfig({
  ADD_TAGS: ALLOWED_CUSTOM_TAGS,
  ADD_ATTR: HTMX_ATTRS,
  RETURN_TRUSTED_TYPE: false,
  CUSTOM_ELEMENT_HANDLING: CUSTOM_ELEMENT_HANDLING,
});

// Create default policy
if (typeof window.trustedTypes !== "undefined") {
  window.trustedTypes.createPolicy("default", {
    createHTML: (html) =>
      DOMPurify.sanitize(html, {
        RETURN_TRUSTED_TYPE: false,
        CUSTOM_ELEMENT_HANDLING: CUSTOM_ELEMENT_HANDLING,
      }),
  });
}
