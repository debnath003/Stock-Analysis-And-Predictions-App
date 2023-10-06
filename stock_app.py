import asyncio
import json
import tkinter as tk
from tkinter import messagebox

import matplotlib.pyplot as plt
import pandas as pd
import websockets
import yfinance as yf
from prophet import Prophet
from tkcalendar import DateEntry

"""
Stock Analysis and Prediction App

This module contains the implementation of a Tkinter-based GUI application for
analyzing and predicting stock market data.

Author: Your Name
"""


class StockApp(tk.Tk):
    """
    A class representing a Stock Market Analysis and Prediction App.
    """

    def __init__(self) -> None:
        """
        Initialize the StockApp.

        This method sets up the main application window and initializes many components.
        """
        super().__init__()

        self.title("Stock Market Analysis and Prediction App")

        # dimensions
        window_width = 350
        window_height = 250
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x_position = (screen_width - window_width) // 2
        y_position = (screen_height - window_height) // 2
        self.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

        self.resizable(width=False, height=False)

        self.portfolio = []  # Portfolio symbols storing

        self.symbol_label = tk.Label(self, text="Enter A Stock Symbol:")
        self.symbol_label.pack()

        self.symbol_entry = tk.Entry(self)
        self.symbol_entry.pack()

        self.add_to_portfolio_button = tk.Button(
            self, text="Add to Portfolio", command=self.add_to_portfolio
        )
        self.add_to_portfolio_button.pack()

        self.view_portfolio_button = tk.Button(
            self, text="View Portfolio", command=self.view_portfolio
        )
        self.view_portfolio_button.pack()

        self.analyze_portfolio_button = tk.Button(
            self, text="Analyze Portfolio", command=self.analyze_portfolio
        )
        self.analyze_portfolio_button.pack()

        self.date_range_frame = tk.Frame(self)
        self.date_range_frame.pack()

        self.start_date_label = tk.Label(self.date_range_frame, text="Start Date:")
        self.start_date_label.pack(side=tk.LEFT)

        self.start_date_entry = DateEntry(self.date_range_frame)
        self.start_date_entry.pack(side=tk.LEFT)

        self.end_date_label = tk.Label(self.date_range_frame, text="End Date:")
        self.end_date_label.pack(side=tk.LEFT)

        self.end_date_entry = DateEntry(self.date_range_frame)
        self.end_date_entry.pack(side=tk.LEFT)

        self.analyze_button = tk.Button(
            self, text="Analyze", command=self.analyze_stock
        )
        self.analyze_button.pack()

        self.predict_button = tk.Button(
            self, text="Predict", command=self.predict_stock
        )
        self.predict_button.pack()

        self.real_time_label = tk.Label(self, text="Real-Time Price: N/A")
        self.real_time_label.pack()

    def analyze_stock(self) -> None:
        """
        Analyze the historical stock data.
        """
        stock_symbol = self.symbol_entry.get().strip().upper()
        start_date = self.start_date_entry.get_date().strftime("%Y-%m-%d")
        end_date = self.end_date_entry.get_date().strftime("%Y-%m-%d")

        if not stock_symbol:
            messagebox.showerror("Error", "Please enter a stock symbol.")
            return

        try:
            self.stock_data = self.fetch_stock_data(stock_symbol, start_date, end_date)
            self.analyze_and_plot(stock_symbol, self.stock_data)
        except yf.errors.YFinanceError as e:  # Catch specific exception
            messagebox.showerror("Error", f"Error fetching data: {e!s}")
            return

    def fetch_stock_data(
        self, symbol: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch historical stock data for the given symbol and date range.
        """
        return yf.download(symbol, start=start_date, end=end_date)

    def analyze_and_plot(self, symbol: str, data: yf.Ticker) -> None:
        """
        Analyze and plot the historical stock data.
        """
        stock_data_df = data.reset_index()[["Date", "Close"]]
        stock_data_df = stock_data_df.rename(columns={"Date": "ds", "Close": "y"})

        plt.figure(figsize=(12, 6))
        plt.plot(stock_data_df["ds"], stock_data_df["y"], label="Historical Prices")
        plt.title(f"{symbol} Stock Price Analysis")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(visible=True)
        plt.show()

    def plot_stock_data(self, symbol: str, data: yf.Ticker) -> None:
        """
        Plot historical stock data.
        """
        date_close = data.reset_index()[["Date", "Close"]]
        date_close = date_close.rename(columns={"Date": "ds", "Close": "y"})

        plt.plot(date_close["ds"], date_close["y"], label=symbol.replace(" ", "_"))

    def analyze_portfolio(self) -> None:
        """
        Analyze the portfolio of stocks.
        """
        if not self.portfolio:
            messagebox.showinfo("Info", "Portfolio is empty.")
            return

        plt.figure(figsize=(12, 6))

        start_date = self.start_date_entry.get_date().strftime("%Y-%m-%d")
        end_date = self.end_date_entry.get_date().strftime("%Y-%m-%d")

        for symbol in self.portfolio:
            try:
                stock_data = self.fetch_stock_data(symbol, start_date, end_date)
                self.plot_stock_data(symbol, stock_data)
            except yf.errors.YFinanceError as e:  # noqa: PERF203
                messagebox.showerror(
                    "Error", f"Error fetching data for {symbol}: {e!s}"
                )

        plt.title("Portfolio Stock Price Analysis")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(visible=True)
        plt.show()

    def predict_stock(self) -> None:
        """
        Predict and plot the stock price.
        """
        if not hasattr(self, "stock_data"):
            messagebox.showerror("Error", "Please fetch data first.")
            return

        symbol = self.symbol_entry.get().strip().upper()
        self.predict_and_plot(symbol, self.stock_data)

    def predict_and_plot(self, symbol: str, data: yf.Ticker) -> None:
        """
        Predict and plot the stock price.
        """
        date_close = data.reset_index()[["Date", "Close"]]
        date_close = date_close.rename(columns={"Date": "ds", "Close": "y"})

        model = Prophet(daily_seasonality=True)
        model.fit(date_close)

        future = model.make_future_dataframe(periods=1825)
        forecast = model.predict(future)

        plt.figure(figsize=(12, 6))
        plt.plot(date_close["ds"], date_close["y"], label="Actual Prices")
        plt.plot(
            forecast["ds"],
            forecast["yhat"],
            label="Predicted Prices",
            linestyle="dashed",
        )
        plt.title(f"{symbol} Stock Price Prediction")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(visible=True)
        plt.show()

    def add_to_portfolio(self) -> None:
        """
        Add a stock symbol to the portfolio.
        """
        stock_symbol = self.symbol_entry.get().strip().upper()
        if stock_symbol not in self.portfolio:
            self.portfolio.append(stock_symbol)
            messagebox.showinfo("Info", f"{stock_symbol} added to portfolio.")
        else:
            messagebox.showinfo("Info", f"{stock_symbol} is already in the portfolio.")

    def view_portfolio(self) -> None:
        """
        View the portfolio of stock symbols.
        """
        portfolio_info = (
            "\n".join(self.portfolio) if self.portfolio else "Portfolio is empty."
        )
        messagebox.showinfo("Portfolio", portfolio_info)

    async def fetch_real_time_data(self, symbol: str) -> None:
        """
        Fetch real-time stock data asynchronously.
        """
        async with websockets.connect(
            f"wss://realtime-stock-api.com/ws/stocks/{symbol}"
        ) as websocket:
            while True:
                data = await websocket.recv()
                stock_data = json.loads(data)
                price = stock_data["price"]
                self.update_real_time_price(price)

    def start_real_time_data(self, symbol: str) -> None:
        """
        Start fetching real-time stock data.
        """
        loop = asyncio.get_event_loop()
        loop.create_task(self.fetch_real_time_data(symbol))

    def update_real_time_price(self, price: str) -> None:
        """
        Update the real-time stock price label.
        """
        self.real_time_label.config(text=f"Real-Time Price: {price} USD")


if __name__ == "__main__":
    app = StockApp()
    app.mainloop()
