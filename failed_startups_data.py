"""
failed_startups_data.py - Recent failed startup anchors.

These rows stabilize the failure class with publicly known shutdowns from
roughly 2019-2025. Funding values are disclosed or commonly reported estimates
and are used for ML signal balance, not financial auditing.
"""

import pandas as pd


FAILED_STARTUPS = [
    {"name": "Airlift", "category_list": "Quick Commerce|Logistics|Transportation", "funding_total_usd": 85000000,
     "status": "closed", "country_code": "PAK", "state_code": "PJ", "region": "Punjab", "city": "Lahore",
     "funding_rounds": 3, "founded_at": "2019-01-01", "first_funding_at": "2019-04-01", "last_funding_at": "2021-08-01"},
    {"name": "Krave Mart", "category_list": "Quick Commerce|Grocery", "funding_total_usd": 6000000,
     "status": "closed", "country_code": "PAK", "state_code": "SD", "region": "Sindh", "city": "Karachi",
     "funding_rounds": 2, "founded_at": "2021-01-01", "first_funding_at": "2021-10-01", "last_funding_at": "2022-06-01"},
    {"name": "Convoy", "category_list": "Logistics|Freight|Transportation", "funding_total_usd": 1100000000,
     "status": "closed", "country_code": "USA", "state_code": "WA", "region": "Seattle", "city": "Seattle",
     "funding_rounds": 12, "founded_at": "2015-01-01", "first_funding_at": "2015-10-01", "last_funding_at": "2022-04-01"},
    {"name": "Zume", "category_list": "FoodTech|Robotics|Packaging", "funding_total_usd": 445000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "Mountain View",
     "funding_rounds": 7, "founded_at": "2015-01-01", "first_funding_at": "2016-01-01", "last_funding_at": "2018-11-01"},
    {"name": "Olive AI", "category_list": "HealthTech|Artificial Intelligence|Automation", "funding_total_usd": 856000000,
     "status": "closed", "country_code": "USA", "state_code": "OH", "region": "Columbus", "city": "Columbus",
     "funding_rounds": 10, "founded_at": "2012-01-01", "first_funding_at": "2013-01-01", "last_funding_at": "2021-07-01"},
    {"name": "Fast", "category_list": "FinTech|Checkout|Payments", "funding_total_usd": 124000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 4, "founded_at": "2019-01-01", "first_funding_at": "2019-10-01", "last_funding_at": "2021-01-01"},
    {"name": "IRL", "category_list": "Social Network|Messaging|Events", "funding_total_usd": 200000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 5, "founded_at": "2017-01-01", "first_funding_at": "2018-01-01", "last_funding_at": "2021-06-01"},
    {"name": "Veev", "category_list": "Construction|Real Estate|Hardware", "funding_total_usd": 600000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Mateo",
     "funding_rounds": 8, "founded_at": "2008-01-01", "first_funding_at": "2014-01-01", "last_funding_at": "2022-03-01"},
    {"name": "Katerra", "category_list": "Construction|Real Estate|Supply Chain", "funding_total_usd": 1600000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "Menlo Park",
     "funding_rounds": 9, "founded_at": "2015-01-01", "first_funding_at": "2016-01-01", "last_funding_at": "2020-05-01"},
    {"name": "Quibi", "category_list": "Media|Entertainment|Streaming", "funding_total_usd": 1750000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "Los Angeles", "city": "Los Angeles",
     "funding_rounds": 4, "founded_at": "2018-01-01", "first_funding_at": "2018-08-01", "last_funding_at": "2020-03-01"},
    {"name": "Koo", "category_list": "Social Media|Messaging", "funding_total_usd": 66000000,
     "status": "closed", "country_code": "IND", "state_code": "KA", "region": "Bangalore", "city": "Bangalore",
     "funding_rounds": 5, "founded_at": "2020-01-01", "first_funding_at": "2020-08-01", "last_funding_at": "2022-02-01"},
    {"name": "ZestMoney", "category_list": "FinTech|Lending|BNPL", "funding_total_usd": 121000000,
     "status": "closed", "country_code": "IND", "state_code": "KA", "region": "Bangalore", "city": "Bangalore",
     "funding_rounds": 8, "founded_at": "2015-01-01", "first_funding_at": "2016-01-01", "last_funding_at": "2022-09-01"},
    {"name": "Sendy", "category_list": "Logistics|B2B Commerce|FMCG", "funding_total_usd": 26500000,
     "status": "closed", "country_code": "KEN", "state_code": "", "region": "Nairobi", "city": "Nairobi",
     "funding_rounds": 6, "founded_at": "2015-01-01", "first_funding_at": "2016-01-01", "last_funding_at": "2020-01-01"},
    {"name": "VanMoof", "category_list": "Hardware|Mobility|E-Bikes", "funding_total_usd": 190000000,
     "status": "closed", "country_code": "NLD", "state_code": "", "region": "Amsterdam", "city": "Amsterdam",
     "funding_rounds": 7, "founded_at": "2009-01-01", "first_funding_at": "2014-01-01", "last_funding_at": "2021-09-01"},
    {"name": "Atrium", "category_list": "LegalTech|SaaS|Professional Services", "funding_total_usd": 75500000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 4, "founded_at": "2017-01-01", "first_funding_at": "2017-06-01", "last_funding_at": "2018-09-01"},
    {"name": "ScaleFactor", "category_list": "FinTech|Accounting|SME SaaS", "funding_total_usd": 103000000,
     "status": "closed", "country_code": "USA", "state_code": "TX", "region": "Austin", "city": "Austin",
     "funding_rounds": 5, "founded_at": "2014-01-01", "first_funding_at": "2017-01-01", "last_funding_at": "2019-08-01"},
    {"name": "Starsky Robotics", "category_list": "Transportation|Autonomous Vehicles|Logistics", "funding_total_usd": 21700000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 4, "founded_at": "2015-01-01", "first_funding_at": "2016-01-01", "last_funding_at": "2018-03-01"},
    {"name": "MoviePass", "category_list": "Entertainment|Subscription|Movies", "funding_total_usd": 68000000,
     "status": "closed", "country_code": "USA", "state_code": "NY", "region": "New York City", "city": "New York",
     "funding_rounds": 5, "founded_at": "2011-01-01", "first_funding_at": "2011-07-01", "last_funding_at": "2018-01-01"},
    {"name": "Homejoy", "category_list": "Marketplace|Home Services", "funding_total_usd": 40000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 4, "founded_at": "2010-01-01", "first_funding_at": "2011-01-01", "last_funding_at": "2013-12-01"},
    {"name": "Sprig", "category_list": "Food Delivery|Mobile|Logistics", "funding_total_usd": 56700000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 5, "founded_at": "2013-01-01", "first_funding_at": "2013-06-01", "last_funding_at": "2016-04-01"},
    {"name": "uBiome", "category_list": "Biotech|HealthTech|Diagnostics", "funding_total_usd": 105000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 6, "founded_at": "2012-01-01", "first_funding_at": "2013-01-01", "last_funding_at": "2018-09-01"},
    {"name": "Braid", "category_list": "FinTech|Payments|Consumer Finance", "funding_total_usd": 10000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 2, "founded_at": "2019-01-01", "first_funding_at": "2021-01-01", "last_funding_at": "2021-08-01"},
    {"name": "Daylight", "category_list": "FinTech|Digital Banking|Consumer Finance", "funding_total_usd": 20000000,
     "status": "closed", "country_code": "USA", "state_code": "NY", "region": "New York City", "city": "New York",
     "funding_rounds": 3, "founded_at": "2020-01-01", "first_funding_at": "2020-08-01", "last_funding_at": "2022-11-01"},
    {"name": "Fuzzy", "category_list": "Pet|HealthTech|Subscription", "funding_total_usd": 80000000,
     "status": "closed", "country_code": "USA", "state_code": "CA", "region": "SF Bay Area", "city": "San Francisco",
     "funding_rounds": 5, "founded_at": "2016-01-01", "first_funding_at": "2017-01-01", "last_funding_at": "2021-10-01"},
    {"name": "Mandolin", "category_list": "Music|Events|Streaming", "funding_total_usd": 17000000,
     "status": "closed", "country_code": "USA", "state_code": "IN", "region": "Indianapolis", "city": "Indianapolis",
     "funding_rounds": 3, "founded_at": "2020-01-01", "first_funding_at": "2020-10-01", "last_funding_at": "2021-06-01"},
]


def get_failed_startups_df():
    df = pd.DataFrame(FAILED_STARTUPS)
    df["permalink"] = "/organization/" + df["name"].str.lower().str.replace(" ", "-", regex=False)
    df["homepage_url"] = ""
    return df
