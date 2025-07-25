import { test, expect } from "@playwright/test";
const { sendMessage, signIn } = require("./utils.js");

test(`Content can be copied to clipboard`, async ({ page }) => {
  await signIn(page);

  await page.goto("/chats");

  await sendMessage(page);

  await page.getByText("Copy information").click();
  const response = page.locator(".chat-message__text").nth(1);

  const messageInput = page.locator(".message-input");
  await messageInput.focus();
  await page.keyboard.press("Meta+V");
  expect(await response.textContent()).toEqual(await messageInput.inputValue());
});
