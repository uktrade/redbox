import { test, expect } from "@playwright/test";
const { signIn } = require("./utils.js");

test(`Chat history functionality works as expected`, async ({ page }) => {
  await signIn(page);

  await page.goto("/chats");

  const count1 = await page.evaluate(
    () => document.querySelectorAll(".chat-item-link").length
  );

  await page.evaluate(() => {
    document.querySelector("chat-history").addChat("session-id", "Test chat");
  });

  // The chat history item has been created
  const chatHistoryItem = page.locator(".chat-item-link").first();
  const count2 = await page.evaluate(
    () => document.querySelectorAll(".chat-item-link").length
  );
  await expect(chatHistoryItem).toContainText("Test chat");
  await expect(chatHistoryItem).toHaveAttribute("href", "/chats/session-id");
  expect(count2).toEqual(count1 + 1);

  // A "Today" heading has been created
  await expect(
    page.locator(".rb-chat-history__date_group").first()
  ).toContainText("Today");

  // A chat can be renamed
  await page.locator(".rb-chat-history__actions-button").first().click();
  await page.locator('button[data-action="rename"]').first().click();
  const textInput = page.locator(".chat-title-text-input input").first();
  await textInput.fill("Renamed chat");
  await textInput.press("Enter");
  await expect(chatHistoryItem).toContainText("Renamed chat");

  // A chat can be deleted
  await page.locator(".rb-chat-history__actions-button").first().click();
  await page.locator('button[data-action="delete"]').first().click();
  await page.locator('button[data-action="delete-confirm"]').first().click();
  await page.waitForTimeout(200);
  const count3 = await page.evaluate(
    () => document.querySelectorAll(".chat-item-link").length
  );
  expect(count3).toEqual(count1);
});
