from flask import Flask, render_template, request, redirect, url_for, session, Response
import pandas as pd
import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)
app.secret_key = "secret123"

data_df = None


# LOGIN 
@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form["username"] == "admin" and request.form["password"] == "admin123":
            session["user"] = "admin"
            return redirect(url_for("dashboard"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


#  DASHBOARD 
@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    global data_df

    if "user" not in session:
        return redirect(url_for("login"))

    #  FILE UPLOAD 
    if request.method == "POST" and request.files.get("file"):

        file = request.files["file"]

        try:
            data_df = pd.read_csv(file, encoding="latin1")

            required = ["Product", "Region", "Sales", "Revenue", "Profit", "Date"]

            for col in required:
                if col not in data_df.columns:
                    return render_template(
                        "dashboard.html",
                        error=f"Missing column: {col}",
                        chart_data=None
                    )

            data_df["Sales"] = pd.to_numeric(data_df["Sales"], errors="coerce")
            data_df["Revenue"] = pd.to_numeric(data_df["Revenue"], errors="coerce")
            data_df["Profit"] = pd.to_numeric(data_df["Profit"], errors="coerce")
            data_df["Date"] = pd.to_datetime(data_df["Date"], errors="coerce")

            data_df = data_df.dropna()

        except Exception as e:
            return render_template("dashboard.html", error=str(e), chart_data=None)

    if data_df is None:
        return render_template("dashboard.html", chart_data=None)

    #  FILTERS 
    df = data_df.copy()

    region = request.form.get("region")
    product = request.form.get("product")

    if region:
        df = df[df["Region"] == region]

    if product:
        df = df[df["Product"] == product]

    #  EMPTY FILTER SAFETY 
    if df.empty:
        return render_template(
            "dashboard.html",
            error="No data found for selected Region + Product.",
            chart_data=None
        )

    #  KPI 
    total_revenue = float(df["Revenue"].sum())
    total_sales = float(df["Sales"].sum())
    total_profit = float(df["Profit"].sum())

    profit_margin = 0 if total_revenue == 0 else round(
        (total_profit / total_revenue) * 100, 2
    )

    product_group = df.groupby("Product")["Sales"].sum()
    region_group = df.groupby("Region")["Revenue"].sum()

    top_product = product_group.idxmax() if not product_group.empty else "N/A"
    top_region = region_group.idxmax() if not region_group.empty else "N/A"

    insight = "Good Profit Zone" if profit_margin > 20 else "Low Profit"
    insight += f". Best Region: {top_region}. Top Product: {top_product}"

    #  ML PREDICTION 
    ml_prediction = "Not enough data"
    ml_accuracy = "Not calculated"

    try:
        df_ml = df.copy()
        df_ml["Date_num"] = df_ml["Date"].map(pd.Timestamp.toordinal)

        X = df_ml[["Date_num"]]
        y = df_ml["Revenue"]

        if len(df_ml) > 5:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=0
            )

            model = LinearRegression()
            model.fit(X_train, y_train)

            pred = model.predict(X_test)
            ml_accuracy = round(mean_absolute_error(y_test, pred), 2)

            future = np.array([[X["Date_num"].max() + 30]])
            ml_prediction = round(model.predict(future)[0], 2)

            pickle.dump(model, open("model.pkl", "wb"))

    except:
        ml_prediction = "ML failed"

    # CHART DATA 
    product_sales = df.groupby("Product")["Sales"].sum()
    region_revenue = df.groupby("Region")["Revenue"].sum()
    profit_trend = df.groupby("Date")["Profit"].sum().sort_index()

    chart_data = {
        "products": product_sales.index.tolist(),
        "product_sales": product_sales.values.tolist(),

        "regions": region_revenue.index.tolist(),
        "region_revenue": region_revenue.values.tolist(),

        "dates": profit_trend.index.astype(str).tolist(),
        "profit_trend": profit_trend.values.tolist()
    }

    return render_template(
        "dashboard.html",
        total_revenue=total_revenue,
        total_sales=total_sales,
        total_profit=total_profit,
        profit_margin=profit_margin,
        insight=insight,
        ml_prediction=ml_prediction,
        ml_accuracy=ml_accuracy,
        chart_data=chart_data
    )


#  DOWNLOAD 
@app.route("/download")
def download():
    global data_df

    if data_df is None:
        return "No data"

    return Response(
        data_df.to_csv(index=False),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=report.csv"}
    )


#  LOGOUT 
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run(debug=True)
