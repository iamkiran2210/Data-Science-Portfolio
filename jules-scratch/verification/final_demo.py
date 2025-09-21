import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            print("Navigating to the application...")
            await page.goto("http://localhost:8501", timeout=120000)

            # Start the chatbot
            await page.get_by_role("button", name="Start Chatbot").click()
            print("Chatbot started.")

            # Wait for the welcome message
            await expect(page.get_by_text("Welcome!")).to_be_visible(timeout=30000)
            print("Welcome message visible.")

            # Send a prompt
            chat_input = page.get_by_role("textbox", name="Ask me anything...")
            await chat_input.fill("hello, what is the capital of France?")
            await chat_input.press("Enter")
            print("Prompt sent.")

            # Wait for assistant response (welcome + user + assistant = 3 messages)
            await expect(page.locator(".stChatMessage")).to_have_count(3, timeout=60000)
            print("Assistant response received.")

            # Take a screenshot
            await page.screenshot(path="jules-scratch/verification/final_demo.png")
            print("Screenshot taken successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            await page.screenshot(path="jules-scratch/verification/error_screenshot.png")

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
