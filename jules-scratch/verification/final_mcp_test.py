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
            continue_button = page.get_by_role("button", name="Continue to Chat")
            await expect(continue_button).to_be_visible(timeout=30000)
            await continue_button.click()
            print("Setup complete, proceeding to chat.")

            # --- Scenario A: Data Query ---
            print("Testing data query...")
            chat_input = page.get_by_role("textbox", name="Ask me anything...")
            await chat_input.fill("data for jaipur")
            await chat_input.press("Enter")

            await expect(page.locator(".stChatMessage")).to_have_count(3, timeout=60000)
            await expect(page.locator(".stChatMessage").nth(2)).to_contain_text("Data for Jaipur")
            print("Data query test passed.")

            # --- Scenario B: LLM Query ---
            print("Testing LLM query...")
            await chat_input.fill("hello")
            await chat_input.press("Enter")

            await expect(page.locator(".stChatMessage")).to_have_count(5, timeout=60000)
            await expect(page.locator(".stChatMessage").nth(4)).not_to_contain_text("Data for")
            print("LLM query test passed.")

            await page.screenshot(path="jules-scratch/verification/final_mcp_convo.png")
            print("Screenshot taken successfully.")

        except Exception as e:
            print(f"An error occurred: {e}")
            await page.screenshot(path="jules-scratch/verification/error_screenshot.png")

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
