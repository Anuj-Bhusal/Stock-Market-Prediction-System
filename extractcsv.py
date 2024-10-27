from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
import pandas as pd
from datetime import datetime, timedelta
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

def get_stock_prices(driver, date):
    url = "https://merolagani.com/StockQuote.aspx"
    driver.get(url)
    
    # Wait for the date input to be present and enter the date
    date_input = WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.ID, "ctl00_ContentPlaceHolder1_txtMarketDateFilter"))
    )
    date_input.clear()
    date_input.send_keys(date)

    # Click the search button
    search_button = driver.find_element(By.ID, "ctl00_ContentPlaceHolder1_lbtnSearch")
    search_button.click()

    try:
        # Wait for the table to be present
        table = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.ID, "ctl00_ContentPlaceHolder1_divData"))
        )
        # Get the table data
        table_html = table.get_attribute('outerHTML')
        df = pd.read_html(StringIO(table_html))[0]
        return df
    except:
        print(f"Table not found for date: {date}")
        return None

def generate_dates(start_date, end_date):
    current_date = start_date
    while current_date <= end_date:
        # Skip Friday (4) and Saturday (5)
        if current_date.weekday() not in [4, 5]:
            yield current_date.strftime("%m/%d/%Y")
        current_date += timedelta(days=1)

def main():
    start_date = datetime.strptime("07/27/2019", "%m/%d/%Y")
    end_date = datetime.now()

    # List of symbols provided in the image
    symbols = ["NTC", "NABIL", "CIT", "NRIC", "GBIME", "NICA", "EBL", "HRL", "NIMB", "SCB", 
               "NLIC", "NIFRA", "UNL", "HIDCL", "HBL", "SHL", "PCBL", "KBL", "ADBL", "SBL", 
               "LSL", "NMB", "SARBTM", "LICN", "SANIMA", "CHCL", "PRVU", "RBCL", "UPPER", "HDL"]

    all_data = []
    valid_dates = []

    for date in generate_dates(start_date, end_date):
        try:
            stock_prices = get_stock_prices(driver, date)
            if stock_prices is not None:
                stock_prices = stock_prices[stock_prices['Symbol'].isin(symbols)]
                stock_prices['Date'] = date  # Add the date to the dataframe
                all_data.append(stock_prices)
                valid_dates.append(date)  # Collect only valid dates
            time.sleep(1)  # Avoid overwhelming the server with requests
        except Exception as e:
            print(f"Error: {e} for date: {date}")

    driver.quit()

    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        tickers = final_df['Symbol'].unique()

        if not os.path.exists('stock_data'):
            os.makedirs('stock_data')

        for ticker in tickers:
            ticker_df = final_df[final_df['Symbol'] == ticker][['Date', 'LTP']]
            ticker_df.to_csv(f"stock_data/{ticker}_stock_prices.csv", index=False)
            print(f"Data transferred to stock_data/{ticker}_stock_prices.csv")

if __name__ == "__main__":
    main()