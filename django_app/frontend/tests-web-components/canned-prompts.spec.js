import { test, expect } from "@playwright/test";
const { signIn } = require("./utils.js");

test(`Clicking canned prompts updates the text input`, async ({ page }) => {
  await signIn(page);

  await page.goto("/chats");

  const textInput = page.locator(".iai-chat-input__input");
  await expect(textInput).toHaveValue("");
});
