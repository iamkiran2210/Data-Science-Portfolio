import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            print("Navigating to the application...")
            await page.goto("http://localhost:8501", timeout=120000)

            # Handle the initial setup screen
            start_button = page.get_by_role("button", name="Start Chatbot")
            await expect(start_button).to_be_visible(timeout=30000)
            await start_button.click()
            print("Chatbot started, proceeding to chat.")

            chat_input = page.get_by_role("textbox", name="Ask me anything about India's groundwater...")
            await expect(chat_input).to_be_visible(timeout=30000)
            print("Chat input is visible.")

            # --- Scenario A: Data Query ---
            print("Testing data query...")
            await chat_input.fill("what is the data for jaipur")
            await chat_input.press("Enter")

            # Wait for messages to appear. The welcome message (1) + user (1) + assistant (1) = 3
            await expect(page.locator(".stChatMessage")).to_have_count(3, timeout=60000)

            # Check the content of the assistant's response
            data_response = page.locator(".stChatMessage").nth(2) # third message
            await expect(data_response).to_contain_text("Data for Jaipur")
            await expect(data_response).to_contain_text("Groundwater Level")
            print("Data query test passed.")

            # --- Scenario B: LLM Query ---
            print("Testing LLM query...")
            await chat_input.fill("hello how are you")
            await chat_input.press("Enter")

            # Wait for 5 messages total now
            await expect(page.locator(".stChatMessage")).to_have_count(5, timeout=60000)

            llm_response = page.locator(".stChatMessage").nth(4) # fifth message
            await expect(llm_response).not_to_contain_text("Data for")
            print("LLM query test passed.")

            # Take a screenshot
            await page.screenshot(path="jules-scratch/verification/mcp_convo.png")
            print("Screenshot taken successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            await page.screenshot(path="jules-scratch/verification/error_screenshot.png")

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
