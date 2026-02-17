import { test, expect } from "@playwright/test";
const { exec } = require("child_process");
require("dotenv").config();

const signIn = async (page) => {
  await page.goto("/sign-in");

  // Perform login actions
  await page.fill("#email", process.env.FROM_EMAIL);
  await page.click('button[type="submit"]');

  await expect(page.locator("h1")).toContainText("Settings");
};

const sendMessage = async (page) => {
  await page.locator(".rbds-message-input").fill("Testing");
  await page.getByRole("button", { name: "Send" }).click();
};

const uploadDocument = async (page) => {
  await page.goto("/upload");
  await page.setInputFiles('input[type="file"]', "./test-upload.html");
  await page.click('button[type="submit"]');
};

module.exports = {
  signIn,
  sendMessage,
  uploadDocument,
};
