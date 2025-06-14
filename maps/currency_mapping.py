country_to_ccy = {
    "Australia":       "AUD",
    "Germany":       "EUR",
    "Spain":"EUR",
    "France":          "EUR",
    "Netherlands":    "EUR",
    "Italy":           "EUR",
    "Portugal":       "EUR",
    "United States":   "USD",
    "New Zealand":     "NZD",
    "Brazil":          "BRL",
    "Canada":          "CAD",
    "Switzerland":     "CHF",
    "Chile":           "CLP",
    "China":           "CNY",
    "Czech Republic":  "CZK",
    "Denmark":         "DKK",
    "Hong Kong":       "HKD",
    "Hungary":         "HUF",
    "Israel":          "ILS",
    "India":           "INR",
    "Japan":           "JPY",
    "South Korea":     "KRW",
    "Mexico":          "MXN",
    "Norway":          "NOK",
    "Pakistan":        "PKR",
    "Poland":          "PLN",
    "Russia":          "RUB",
    "Sweden":          "SEK",
    "Singapore":       "SGD",
    "Turkey":          "TRY",
    "South Africa":    "ZAR",
}

country_to_pair = {
    country: f"GBP{ccy}"
    for country, ccy in country_to_ccy.items()
}
