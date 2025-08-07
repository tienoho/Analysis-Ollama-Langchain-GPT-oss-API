Business Data Analysis and Forecasting API with GPT-OSS, Ollama, LangChain, and SQL Server
Overview of Components and Goals
We aim to build a Python API that combines a local large language model (LLM) with statistical forecasting to analyze business data and predict key metrics (production output, import volumes, revenue) 6–12 months into the future. The solution integrates several components: GPT-OSS (an open-source GPT-based model) running via Ollama (for local LLM serving), LangChain (to orchestrate LLM and tools), Microsoft SQL Server (housing business data), and time-series models like ARIMA and Prophet for forecasts. The API will accept queries (possibly in natural language) about business metrics, automatically query the SQL database for relevant historical data, use both LLM reasoning and statistical models to generate forecasts (monthly, quarterly, or yearly as needed), and return the results as JSON. This approach leverages the strengths of both LLMs and traditional forecasting models to provide insightful and accurate business predictions.
GPT-OSS and Ollama for Local LLM Reasoning
Ollama is a system that allows running open-source large language models (LLMs) locally, bundling the model weights and configuration for easy deployment
python.langchain.com
. In our setup, we can use an open-source GPT-like model (GPT-OSS) loaded into Ollama to serve as the AI engine. This local LLM avoids external API calls, ensuring data privacy and reducing costs, since all inference happens on our hardware. We can interface with the model through LangChain's Ollama integration. For example, the langchain-ollama package provides a ChatOllama class that lets us invoke a locally hosted model by name
medium.com
. Using this, the API can send prompts or questions to the LLM and get back responses. LangChain serves as the framework that makes the LLM more functional and integrates it with other tools. As one source puts it, "LangChain is a framework for building applications that leverage AI and LLMs. It simplifies the development of complex AI applications by making LLMs more functional."
medium.com
 In this project, LangChain will allow the GPT-OSS model (via Ollama) to interact with the SQL database and perform computations. The LLM can be prompted not only with the user’s query, but also with instructions or context (like the database schema or example queries) so that it can produce useful outputs such as SQL statements or explanations.
Architecture: LLM-Driven SQL Querying and Forecasting
The overall flow of a query through the system will be as follows
python.langchain.com
:
Convert user query to SQL: The user (or client application) asks a question or requests a forecast (e.g., "Forecast next year's revenue by quarter"). The LLM (via LangChain) interprets this request and generates an appropriate SQL query to retrieve the historical data needed to answer it
python.langchain.com
. For instance, it might produce a SQL query that sums monthly revenue from the sales orders tables for the past few years.
Execute SQL query: The generated SQL is executed against the Microsoft SQL Server database. LangChain provides a SQLDatabase utility (wrapping SQLAlchemy) to connect to the database and run queries. We would connect using a connection string that includes the SQL Server driver. For example:
python
Sao chép
uri = "mssql+pyodbc://DBUsername:DBPassword@ServerName:Port/DBName?driver=ODBC+Driver+17+for+SQL+Server"
db = SQLDatabase.from_uri(uri)
This uses the Microsoft ODBC Driver 17 for SQL Server via pyodbc
stackoverflow.com
. With a proper connection, the LLM (through LangChain's agent or chain) can issue SELECT queries to fetch data.
Retrieve data and generate answer/forecast: The query results (e.g. a time series of monthly sales or production quantities) are then used to generate the forecast. Here is where we combine the LLM with statistical models:
The LLM can analyze the query results and decide on a forecasting approach or even directly produce a forecast narrative. However, LLMs alone are not specialized for precise numeric time-series projection, so we integrate statistical forecasting models for accuracy.
We apply ARIMA or Prophet on the historical data to project future values. This can be done by calling these models in Python code (which can be a part of the LangChain workflow or a separate function that the chain/agent uses as a tool). For example, using Prophet: we prepare a dataframe of dates and values from the SQL output, fit a Prophet model, and produce a 6–12 month forecast. With ARIMA, we could use an automated ARIMA (such as pmdarima.auto_arima) to fit the best model and forecast forward.
LLM augmentation (if needed): The LLM can optionally be used to provide context or adjust the forecast. For instance, the LLM might incorporate domain knowledge (holidays, promotions, anomalies) or at least format the output. Modern LLMs can bring in context that pure time-series algorithms might miss — e.g., recognizing that a dip in data was due to a one-off event and adjusting accordingly. LLMs handle messy data (missing values, outliers) more gracefully by reasoning in context
dzone.com
, and they "bring context to the table," knowing patterns like "retail sales spike during holidays or energy consumption changes with seasons" from their broad training
dzone.com
. This synergy between LLM and stats can improve accuracy; a real-world case saw a 31% reduction in forecast error by combining LLM insights with traditional models
dzone.com
.
Return JSON result: Finally, the system returns the forecast result as a JSON response via the API. The JSON could include the predicted values for each future period and possibly metadata (like the model used, confidence intervals, etc.). For example:
json
Sao chép
{
  "metric": "Revenue",
  "forecast_horizon": "12 months",
  "forecast_unit": "USD",
  "method": "Prophet",
  "predictions": [
    {"period": "2024-Q1", "forecast": 1250000},
    {"period": "2024-Q2", "forecast": 1320000},
    ...
  ]
}
This JSON structure is easily consumable by client applications. The API focuses on returning data (numbers and labels) rather than lengthy text, since the primary output is meant for further processing or visualization.
Throughout this process, LangChain acts as the glue: it can use an Agent with tools or a specialized SQL chain to manage steps. The LLM is guided by prompt templates that include the database schema and examples, ensuring it writes correct SQL. According to LangChain’s docs, enabling an LLM to query structured data often means letting the model "write and execute queries in a DSL, such as SQL"
python.langchain.com
python.langchain.com
. The agent might iterate: for complex questions it could do multiple queries and computations until it has the data needed to answer.
Microsoft SQL Server Data Integration
The business data resides in a Microsoft SQL Server database with a known schema (tables for delivery notes, sales orders, inventory, purchase orders, etc., as described). We will integrate this DB into our pipeline securely and efficiently:
Connection: We use SQLAlchemy (via LangChain's SQLDatabase) to connect. As shown earlier, a correct ODBC connection string is key. In our environment, we must have an appropriate ODBC driver (e.g., "ODBC Driver 17 for SQL Server") installed. The Stack Overflow example above demonstrates how to format the connection URI
stackoverflow.com
. Another approach is using a DSN-less connection string with odbc_connect parameters
stackoverflow.com
, which can be useful especially on Linux environments with unixODBC. The result is an SQL engine the LLM agent can query. We will ensure this DB user has read-only access to the necessary tables to mitigate any risk of harmful queries
python.langchain.com
.
Schema awareness: The provided database structure outlines tables like Orhkrg (delivery note header), Orhsrg (delivery line items), Orkrg (order header), Orsrg (order lines), Voorrd (stock levels per warehouse), etc. We can feed this schema info into the LLM’s prompt or use LangChain’s toolkits which automatically extract table schema. For instance, db.get_usable_table_names() and introspecting column names helps the LLM know what it can query
python.langchain.com
. The LLM might receive a system message like: "You are connected to a database with tables Orkrg (sales orders), Orsrg (order lines), ... Orsrg has fields ArtCode, Esr_aantal (quantity), Extra_pr (net amount) ..." etc. This way, if the user asks "predict next 12 months revenue", the LLM knows it should sum Orsrg.Extra_pr (sales line amounts) by month from past data, or if asked about "production output", it might use Orhsrg (delivery quantities) as a proxy for production volume.
Example: Suppose the user requests a forecast of monthly production output. The LLM could generate a SQL query joining Orhkrg and Orhsrg (delivery notes header and lines) to get all delivered quantities of finished products per month for the last few years. Similarly, for revenue, it could sum the Extra_pr from Orsrg (sales order lines, excluding VAT) grouped by month and year. For import volumes, it might look at purchase orders (perhaps using Orkrg where order type indicates purchase, or the ledger transactions in Gbkmut with TransType/TransSubType indicating received goods). Once the SQL is generated, LangChain executes it and returns the data (e.g. a list of month-year and quantity or amount). This data then feeds the forecasting step.
By automating the SQL querying, we ensure the pipeline is dynamic – whenever the API is asked for a forecast on a different product or scenario, it will pull the latest data from SQL Server without manual intervention. This auto-querying from SQL is central to keeping the analysis up-to-date.
Time-Series Forecasting with ARIMA and Prophet
For the core forecasting task, we incorporate well-established time series models:
ARIMA (AutoRegressive Integrated Moving Average): ARIMA models capture patterns based on past values (autoregression) and past errors (moving average), after differencing the series to achieve stationarity. The model is specified by parameters (p, d, q) for AR, differencing, and MA terms. In practice, using an automated approach like pmdarima.auto_arima can streamline the process by testing different combinations of p, d, q to find an optimal model
neptune.ai
. ARIMA is powerful and, in a controlled experiment, it achieved the smallest prediction error on certain stock data
neptune.ai
. However, ARIMA requires that the user understand the time series properties (trend, seasonality) and may need careful tuning and domain knowledge
neptune.ai
. It works best for datasets where patterns change slowly and there's sufficient history; one must also ensure the data is stationary (which might involve transformations).
Prophet: Prophet (developed by Facebook/Meta) is specifically designed for business time series forecasting, intended to be easy to use and automate the detection of common patterns (trend, yearly seasonality, weekly seasonality, holidays). It often requires less manual tuning than ARIMA because it has built-in assumptions for typical business data patterns
neptune.ai
. Prophet excels at series with clear seasonal effects and a strong temporal structure (like sales that peak on holidays, or traffic that has weekly cycles). It will, however, create a model that could be less accurate if the data doesn't conform to those patterns (and Prophet always requires a date/time index — if the data points are irregular or not calendar-based, it might struggle)
neptune.ai
.
In our API, we can use both depending on the scenario. For example, if forecasting demand for a product with strong seasonal trends, Prophet might be a quick solution. If fine-tuned control or non-seasonal subtle patterns are needed, ARIMA might perform better (at the cost of more computation to find the right parameters). In fact, a comparative study noted: "ARIMA is a powerful model... but might need careful hyperparameter tuning and a good understanding of the data, whereas Prophet requires less tuning as it is designed to detect patterns in business time series."
neptune.ai
 We can also incorporate both models and compare or ensemble their results for robustness. The API could allow specifying the model (method="ARIMA" or "Prophet") or choose automatically. If using ARIMA, the implementation might involve taking the SQL data (time series), using statsmodels or pmdarima to fit an ARIMA. If using Prophet, we use the Prophet library to fit/predict. Both can output future values for each period. Forecast Horizon and Granularity: The user mentioned forecasting by time period (month/quarter/year) and 6–12 months ahead. Our API should be flexible in this regard. We can design the system to resample or aggregate data to the desired granularity before forecasting. For monthly forecasts, we ensure the historical data is aggregated monthly. For quarterly, aggregate to quarters, etc. Prophet can handle different frequencies as long as the date field reflects it (or we aggregate manually). ARIMA can operate on any evenly spaced interval (monthly, quarterly) as well. The API could accept a parameter like freq="M" or "Q" and horizon=12 to mean 12 future periods of that frequency. Internally, we'd prepare data accordingly.
Combining LLM Insight with Statistical Models
One innovative aspect of this system is using the LLM not just to fetch data but to enhance forecasting. LLMs have been "shaking things up" in time-series analysis
dzone.com
. They can recognize complex patterns without explicit programming because they've seen many sequences of data during training (even if as text or other contexts). In our context, the LLM could assist in a few ways:
Intelligent model selection: The LLM could examine the historical data (it can be given a summary of it, like trends or anomalies) and decide whether ARIMA or Prophet (or even a combination) is more suitable. It might notice seasonal language in data or other hints.
Feature augmentation: We can prompt the LLM to identify external factors or qualitative features that might affect the forecast. For example, if we feed it the time series of sales and maybe some related info (marketing events, etc.), it might output that certain spikes coincide with events (like "the spike in Dec is due to holiday sales"). This could be used to adjust the forecast or at least inform us.
Handling outliers and missing data: As noted, LLMs "handle messy data better. Missing values? Outliers? LLMs can work around these issues more gracefully than traditional methods."
dzone.com
 The LLM could flag or fill missing periods in a reasonable way before modeling. It could also downplay outliers by explaining them (e.g., ignoring an extreme one-off sale in the projections).
Narrative output: Although the API returns JSON, the LLM could also generate a human-readable explanation of the forecast for reporting purposes (this could be an extended feature where the API returns both data and an AI-generated summary). For instance, "Sales are projected to grow 5% each quarter next year, with a peak in Q4, likely due to seasonal holiday demand." This uses the LLM's natural language ability to communicate the numbers.
Recent experiments in combining LLMs with time-series data report substantial improvements. One source describes a manufacturing company that "reduced forecast error by 31% compared to their existing ARIMA models" by using an LLM-driven approach
dzone.com
. The LLM brings in context (like knowing about promotions or external economic indicators) that a pure ARIMA model wouldn’t have. Our system can leverage this by allowing the LLM to adjust the statistical forecast. Concretely, we might generate an initial forecast with Prophet/ARIMA, then feed those results back into the LLM (along with historical data) and ask if it foresees any adjustments or if it can fine-tune the numbers based on its general knowledge. However, this must be done cautiously to avoid introducing spurious "gut feelings" from the LLM that aren't grounded in data. One strategy is to let the LLM handle qualitative adjustments (like timing of peaks) while the quantitative base comes from the statistical model. In summary, the LLM and the statistical models complement each other: the stats models ensure rigorous numerical predictions grounded in historical patterns, while the LLM provides flexibility, contextual understanding, and the ability to parse natural-language queries and instructions.
API Implementation (Python, FastAPI)
To expose this functionality, we'll create a RESTful API (for example, using FastAPI in Python). FastAPI makes it straightforward to define endpoints that execute our LangChain pipeline and return JSON. Each endpoint can correspond to a certain type of query or analysis, or a single endpoint could handle all requests by reading the query parameters or JSON payload. A possible design is an endpoint like POST /forecast where the client sends a request specifying what to forecast. For instance, a JSON payload: {"query": "forecast revenue for next 4 quarters"} or a structured request like {"metric": "Revenue", "horizon": 4, "frequency": "Q"}. The API handler would then:
Instantiate or reuse a LangChain agent/chain that has the Ollama LLM and SQL tool.
If the input is natural language, pass the query to the chain (which will involve the LLM generating SQL, retrieving data, etc., as described).
If the input is structured (e.g., specific metric and horizon), the code might directly formulate the SQL (bypassing the LLM for that step) and only use the LLM for optional narrative. Either approach is possible; using the LLM even for translating structured requests adds flexibility (the LLM could optimize or add insights), but direct SQL might be simpler for predefined metrics.
Run the forecasting logic: call ARIMA/Prophet, possibly involve LLM for context adjustments.
Format the results as a JSON and return.
FastAPI automatically serializes Python dictionaries to JSON, so if our final result is a Python dict (like the example earlier), it will be returned as JSON. We should also handle errors (e.g., if the query is not understood or data is insufficient) and return appropriate HTTP status codes or error messages in JSON. To implement the LangChain part inside FastAPI, we can initialize it at startup (loading the model into memory via Ollama) so that each request doesn’t incur model load time. The Ollama model server will likely be running as a background service (Ollama can be started with ollama serve on port 11434 by default
python.langchain.com
, and the LangChain Ollama client will connect to that). Optionally, we can use LangServe – an extension of LangChain for deploying chains as an API. LangServe is built on FastAPI and Pydantic, and it can expose LangChain chains with automatically generated endpoints/docs
python.langchain.com
. For example, we could wrap our entire chain (LLM + SQL + forecast) as a single LangChain Runnable or chain, and LangServe would handle the web server part. This could accelerate development, as "LangServe helps developers deploy LangChain runnables and chains as a REST API" and is tightly integrated with FastAPI
python.langchain.com
. However, using LangServe is optional; a custom FastAPI app is also fine and might be more straightforward to customize.
Security and Performance Considerations
Because this system allows executing dynamic SQL, we must be mindful of SQL injection or harmful queries. Using an LLM to generate SQL means we should validate the SQL (ensuring it’s read-only and within allowed tables). We configure the database user with minimal permissions (read-only on specific schemas). LangChain's documentation also emphasizes scoping the DB user permissions as a safety measure
python.langchain.com
. We may also sanitize or review the LLM-generated query before execution. Performance-wise, heavy operations are the LLM inference and the forecasting model fitting. Running a 7B or 13B parameter model on a local machine might be slower than an API call to OpenAI, but with proper hardware (GPU or optimized CPU inference with quantization) and given our queries are not continuous high volume, this should be workable. We can further optimize by caching results for repeated queries (e.g., if the same forecast is requested often, cache the result for some time).
Example Workflow
To make this concrete, let's walk through an example request to the API:
User input: {"query": "Dự báo sản lượng xuất kho theo tháng cho 2024"} (Vietnamese for "Forecast the monthly delivered production for 2024"). The API passes this query to the LangChain agent.
LLM Processing: The agent prompt includes the schema. The LLM sees keywords "sản lượng xuất kho" (output volume delivered) and "theo tháng cho 2024". It knows from schema that "xuất kho" (delivery notes) are in Orhkrg/Orhsrg tables. It might produce SQL like:
sql
Sao chép
SELECT YEAR(Pakbon_dat) AS Year, MONTH(Pakbon_dat) AS Month, SUM(quantity) AS TotalOutput
FROM Orhkrg 
JOIN Orhsrg ON Orhsrg.Pakbon_Nr = Orhkrg.Pakbon_Nr
WHERE YEAR(Pakbon_dat) >= 2018
GROUP BY YEAR(Pakbon_dat), MONTH(Pakbon_dat)
ORDER BY Year, Month;
(Here quantity is a placeholder for the actual quantity field in Orhsrg; we'd use the real column, perhaps Esr_aantal if similar to order lines.)
Data retrieval: The above query runs, returns monthly totals of output for past years.
Forecasting: The agent (or our code) takes the monthly series and uses Prophet to forecast 2024 monthly output. Prophet would fit trend and yearly seasonality on 2018–2023 data and predict Jan–Dec 2024. The LLM might be prompted with the raw forecast to check if it wants to adjust (for example, if 2024 includes a known event, the LLM could suggest an adjustment, though that would require giving it that info).
Output: The API returns JSON like:
json
Sao chép
{
  "metric": "Production Output",
  "frequency": "monthly",
  "forecast_horizon": 12,
  "unit": "units",
  "predictions": [
     {"period": "2024-01", "forecast": 5020},
     {"period": "2024-02", "forecast": 4901},
     ...
     {"period": "2024-12", "forecast": 5300}
  ]
}
This JSON can then be used by the caller to display a chart or further analysis. The API could also include the historical data or confidence intervals if needed.
Conclusion
By combining a local GPT-powered LLM with LangChain and SQL integration, and augmenting it with ARIMA/Prophet forecasting, we get the best of both worlds: natural language understanding and contextual reasoning from the LLM, and rigorous time-series predictions from statistical models. This system can automatically fetch relevant business data from SQL Server, analyze trends, and forecast future metrics in a flexible yet accurate manner. Early results and literature suggest that such hybrid approaches can outperform purely statistical forecasts, thanks to the LLM’s ability to incorporate domain knowledge and handle irregularities
dzone.com
dzone.com
. The end result is an API that can answer complex business questions (like "What is our expected revenue next year by quarter?" or "How much inventory should we plan to import in the next 6 months?") with data-backed answers, delivered in a machine-friendly JSON format for easy integration into dashboards or applications. With Ollama managing the GPT-OSS model locally, we ensure data remains on-premise. LangChain provides the scaffold to link the model with database actions and computations. SQL Server provides the authoritative data source. And Prophet/ARIMA inject domain-specific forecasting strength. This aligns well with the goal of a modern AI-assisted business intelligence system: leveraging AI capabilities while grounded in reliable data and proven algorithms. All components are modular, so the system can be extended (for example, adding anomaly detection, what-if scenario analysis, or incorporating other ML models) as the business needs evolve.
