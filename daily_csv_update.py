from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime
from io import StringIO
import time
import os

# Configure ChromeOptions to use Brave
chrome_options = Options()
chrome_options.binary_location = "C:\\Program Files\\BraveSoftware\\Brave-Browser\\Application\\brave.exe"

# Set up the ChromeDriver service
chrome_service = ChromeService(executable_path="C:\\WebDriver\\bin\\chromedriver.exe")

# Initialize the WebDriver
driver = webdriver.Chrome(service=chrome_service, options=chrome_options)

def get_latest_stock_prices(driver):
    url = "https://merolagani.com/StockQuote.aspx"
    driver.get(url)
    
    try:
        # Wait for the table to be present
        table = WebDriverWait(driver, 4).until(
            EC.presence_of_element_located((By.ID, "ctl00_ContentPlaceHolder1_divData"))
        )
        # Get the table data
        table_html = table.get_attribute('outerHTML')
        df = pd.read_html(StringIO(table_html))[0]
        return df
    except:
        print("Table not found for latest date")
        return None

def append_latest_data():
    # List of symbols provided in the image
    symbols = ["NTC", "NABIL", "CIT", "NRIC", "GBIME", "NICA", "EBL", "HRL", "NIMB", "SCB", 
               "NLIC", "NIFRA", "UNL", "HIDCL", "HBL", "SHL", "PCBL", "KBL", "ADBL", "SBL", 
               "LSL", "NMB", "SARBTM", "LICN", "SANIMA", "CHCL", "PRVU", "RBCL", "UPPER", "HDL"]

    latest_date = datetime.now().strftime("%m/%d/%Y")
    latest_prices = get_latest_stock_prices(driver)

    driver.quit()

    if latest_prices is not None:
        latest_prices = latest_prices[latest_prices['Symbol'].isin(symbols)]
        latest_prices['Date'] = latest_date

        for symbol in symbols:
            ticker_df = latest_prices[latest_prices['Symbol'] == symbol][['Date', 'LTP']]
            file_path = f'stock_data/{symbol}_stock_prices.csv'
            
            if os.path.exists(file_path):
                existing_df = pd.read_csv(file_path)
                updated_df = pd.concat([existing_df, ticker_df], ignore_index=True)
                updated_df.to_csv(file_path, index=False)
                print(f"Appended latest data to {file_path}")
            else:
                print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    append_latest_data()
