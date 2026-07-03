const form = document.querySelector("#prediction-form");
const stockSelect = document.querySelector("#stock");
const statusBox = document.querySelector("#status");
const predicted = document.querySelector("#predicted");
const change = document.querySelector("#change");
const changePercent = document.querySelector("#change-percent");
const metadata = document.querySelector("#metadata");

const formatPrice = (value) =>
  Number(value).toLocaleString(undefined, { maximumFractionDigits: 4 });

async function loadStocks() {
  const response = await fetch("/api/stocks");
  const stocks = await response.json();
  stockSelect.innerHTML = stocks.map((stock) => `<option value="${stock}">${stock}</option>`).join("");
}

async function loadMetadata() {
  const response = await fetch("/api/metadata");
  const data = await response.json();
  const metrics = data.metrics || {};
  metadata.textContent = `Rows: ${data.rows || "--"} | Stocks: ${data.stocks || "--"} | RMSE: ${
    metrics.rmse || "--"
  } | R2: ${metrics.r2 || "--"}`;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = form.querySelector("button");
  button.disabled = true;
  statusBox.textContent = "Calculating forecast...";

  const payload = Object.fromEntries(new FormData(form).entries());
  for (const key of ["open_price", "high_price", "low_price", "close_price", "volume"]) {
    payload[key] = Number(payload[key]);
  }

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Prediction failed.");
    }
    predicted.textContent = formatPrice(data.predicted_close_price);
    change.textContent = formatPrice(data.expected_change);
    changePercent.textContent = `${formatPrice(data.expected_change_percent)}%`;
    statusBox.textContent = `${data.stock} forecast generated from the submitted market values.`;
  } catch (error) {
    statusBox.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

loadStocks().catch(() => {
  statusBox.textContent = "Could not load stock symbols.";
});
loadMetadata().catch(() => {
  metadata.textContent = "Model metadata unavailable.";
});
