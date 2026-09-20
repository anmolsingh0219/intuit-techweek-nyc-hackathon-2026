# Feature Lab - WOE / IV scorecard view of every feature

Labeled training book (prior-approved & matured): **51,722 rows**, default rate **0.174**. Validation labeled: **2,551 rows**, default rate **0.206** (the upward drift).

WOE = ln(%non-default / %default); **positive WOE = safer bin**. IV is the feature's predictive *weight*: <0.02 useless, 0.02-0.1 weak, 0.1-0.3 medium, 0.3-0.5 strong, >0.5 suspicious (leakage/selection).

## 1. Feature weights (Information Value), ranked

| feature | group | type | null_pct | n_unique | IV | IV_band | IV_val | PSI | uni_AUC | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| invoice_payment_delinquency_rate | platform_engagement | numeric | 0.0 | 51722 | 0.8573 | suspicious | 0.7109 | 0.014 | 0.7506 | monotone |
| ENG:leverage_x_delinq | engineered | numeric | 0.0 | 51722 | 0.8286 | suspicious | 0.7502 | 0.03 | 0.7476 |  |
| ENG:util_sq | engineered | numeric | 0.0 | 51722 | 0.7368 | suspicious | 0.6326 | 0.005 | 0.7353 |  |
| aggregate_credit_utilization | bureau_credit | numeric | 0.0 | 51722 | 0.7368 | suspicious | 0.6326 | 0.005 | 0.7353 | monotone |
| ENG:affordability_x_delinq | engineered | numeric | 32.8 | 34751 | 0.6508 | suspicious | 0.6577 | 0.014 | 0.6937 |  |
| observed_cash_balance_p10 | bank_feed | numeric | 32.8 | 34751 | 0.5504 | suspicious | 0.5271 | 0.012 | 0.6789 | monotone |
| ENG:cash_runway_days | engineered | numeric | 32.8 | 34751 | 0.5082 | suspicious | 0.5631 | 0.112 | 0.67 |  |
| ENG:score_x_util | engineered | numeric | 0.0 | 51722 | 0.3741 | strong | 0.4273 | 0.011 | 0.6671 |  |
| requested_amount_to_observed_revenue | application_context | numeric | 0.0 | 51722 | 0.3584 | strong | 0.3655 | 0.226 | 0.6675 | monotone |
| ENG:draw_to_daily_rev | engineered | numeric | 32.8 | 34751 | 0.2951 | medium | 0.2921 | 0.147 | 0.6351 |  |
| ENG:loan_to_observed_annual_rev | engineered | numeric | 32.8 | 34751 | 0.2951 | medium | 0.2921 | 0.147 | 0.6351 |  |
| ENG:draw_to_cash_buffer | engineered | numeric | 32.8 | 34751 | 0.285 | medium | 0.3182 | 0.001 | 0.6016 |  |
| requested_amount | self_reported | numeric | 0.0 | 51722 | 0.2791 | medium | 0.2717 | 0.393 | 0.6491 | monotone |
| prior_approved_amount | prior_underwriter | numeric | 0.0 | 51722 | 0.2463 | medium | 0.2487 | 0.358 | 0.6402 | monotone |
| ENG:req_to_stated_rev | engineered | numeric | 0.0 | 51722 | 0.2186 | medium | 0.2094 | 0.268 | 0.6322 |  |
| payroll_regularity_score | bank_feed | numeric | 32.8 | 34698 | 0.1997 | medium | 0.1903 | 0.004 | 0.6114 | monotone |
| observed_revenue_volatility | bank_feed | numeric | 32.8 | 34751 | 0.1846 | medium | 0.187 | 0.025 | 0.6082 | monotone |
| owner_personal_credit_band | bureau_credit | ordinal | 0.0 | 5 | 0.1822 | medium | 0.1953 | 0.008 | nan | rank_corr=-1.00 |
| ENG:stated_vs_observed_rev | engineered | numeric | 32.8 | 34751 | 0.177 | medium | 0.162 | 0.005 | 0.6056 |  |
| ENG:util_x_inquiries | engineered | numeric | 0.0 | 20658 | 0.16 | medium | 0.1239 | 0.003 | 0.5732 |  |
| existing_debt_obligations | bureau_credit | numeric | 0.0 | 51722 | 0.1408 | medium | 0.1747 | 0.111 | 0.6066 | monotone |
| observed_overdraft_count_3mo | bank_feed | numeric | 32.8 | 7 | 0.11 | medium | 0.1145 | 0.004 | 0.5595 | monotone |
| ENG:overdraft_per_month | engineered | numeric | 32.8 | 7 | 0.11 | medium | 0.1145 | 0.004 | 0.5595 |  |
| ENG:debt_service_ratio | engineered | numeric | 32.8 | 34751 | 0.1076 | medium | 0.1311 | 0.231 | 0.581 |  |
| ENG:score_sq | engineered | numeric | 0.0 | 51722 | 0.0988 | weak | 0.059 | 0.014 | 0.5868 |  |
| prior_underwriter_score | prior_underwriter | numeric | 0.0 | 51722 | 0.0988 | weak | 0.059 | 0.014 | 0.5868 | monotone |
| bookkeeping_recency_days | platform_engagement | numeric | 0.0 | 51722 | 0.0952 | weak | 0.143 | 0.006 | 0.586 | monotone |
| days_since_last_inquiry_elsewhere | application_context | numeric | 54.5 | 23558 | 0.0874 | weak | 0.0894 | 0.004 | 0.5416 | monotone |
| ENG:debt_to_stated_rev | engineered | numeric | 0.0 | 51722 | 0.0845 | weak | 0.1148 | 0.373 | 0.5833 |  |
| days_since_last_external_decline | bureau_credit | numeric | 54.8 | 23388 | 0.0783 | weak | 0.0326 | 0.002 | 0.5347 | monotone |
| observed_monthly_revenue_avg_3mo | bank_feed | numeric | 32.8 | 34751 | 0.0718 | weak | 0.0899 | 0.294 | 0.5661 | monotone |
| ENG:revenue_signal | engineered | numeric | 32.8 | 34751 | 0.0708 | weak | 0.0154 | 0.724 | 0.5512 |  |
| ENG:util_x_band | engineered | numeric | 0.0 | 44806 | 0.0689 | weak | 0.0697 | 0.002 | 0.5383 |  |
| platform_active_months | platform_engagement | numeric | 0.0 | 27 | 0.0684 | weak | 0.1115 | 0.006 | 0.5748 | monotone |
| vintage_years | business_identity | numeric | 0.0 | 44979 | 0.0675 | weak | 0.0622 | 0.006 | 0.5699 | monotone |
| employee_count_bucket | business_identity | ordinal | 0.0 | 4 | 0.0553 | weak | 0.0429 | 0.0 | nan | rank_corr=-1.00 |
| stated_time_in_business | self_reported | numeric | 0.0 | 51055 | 0.0533 | weak | 0.0372 | 0.005 | 0.559 | monotone |
| observed_revenue_trend_3mo | bank_feed | numeric | 32.8 | 34751 | 0.0508 | weak | 0.0643 | 1.589 | 0.5542 | monotone |
| sector | business_identity | nominal | 0.0 | 5 | 0.0483 | weak | 0.0429 | 0.001 | nan |  |
| account_age_days | platform_engagement | numeric | 0.0 | 4020 | 0.0469 | weak | 0.0235 | 0.005 | 0.557 | monotone |
| ENG:inquiry_intensity | engineered | numeric | 0.0 | 8 | 0.0391 | weak | 0.0382 | 0.001 | 0.5548 |  |
| recent_inquiries_count_6mo | bureau_credit | numeric | 0.0 | 6 | 0.0313 | weak | 0.0152 | 0.0 | 0.5448 | monotone |
| stated_annual_revenue | self_reported | numeric | 0.0 | 51722 | 0.0226 | weak | 0.0339 | 0.454 | 0.5424 | monotone |
| prior_loans_amount_total | platform_engagement | numeric | 0.0 | 17469 | 0.0223 | weak | 0.0248 | 0.008 | 0.5231 | monotone |
| multi_lender_inquiry_count_30d | application_context | numeric | 0.0 | 6 | 0.0101 | useless | 0.0327 | 0.0 | 0.5258 | monotone |
| has_linked_bank_feed | bank_feed | binary | 0.0 | 2 | 0.0061 | useless | 0.0 | 0.0 | nan |  |
| ENG:prior_default_rate | engineered | numeric | 66.2 | 6 | 0.0059 | useless | 0.0088 | 0.004 | 0.5027 |  |
| prior_loans_count | platform_engagement | numeric | 0.0 | 6 | 0.0037 | useless | 0.0088 | 0.004 | 0.5151 | monotone |
| application_channel | application_context | nominal | 0.0 | 3 | 0.0018 | useless | 0.0012 | 0.0 | nan |  |
| prior_loans_default_count | platform_engagement | numeric | 0.0 | 3 | 0.0007 | useless | nan | nan | 0.5026 | n/a |
| intended_use_of_funds | self_reported | nominal | 0.0 | 4 | 0.0006 | useless | 0.0152 | 0.001 | nan |  |
| geography_region | business_identity | nominal | 0.0 | 4 | 0.0004 | useless | 0.0103 | 0.0 | nan |  |
| repeat_application_count | application_context | numeric | 0.0 | 4 | 0.0001 | useless | 0.0005 | 0.005 | 0.5023 | monotone |


## 2. WOE / monotonic bin tables for the strongest features


### invoice_payment_delinquency_rate  (IV=0.8573)

_Rate of late invoice payments (0-1)._


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.05018) | 5173 | 210 | 0.0406 | 1.607 | 0.1493 |
| 1:[0.05018,0.07353) | 5172 | 296 | 0.0572 | 1.246 | 0.1014 |
| 2:[0.07353,0.09613) | 5172 | 373 | 0.0721 | 1.0 | 0.071 |
| 3:[0.09613,0.1201) | 5172 | 468 | 0.0905 | 0.753 | 0.0439 |
| 4:[0.1201,0.1474) | 5172 | 577 | 0.1116 | 0.52 | 0.0227 |
| 5:[0.1474,0.178) | 5172 | 714 | 0.1381 | 0.277 | 0.007 |
| 6:[0.178,0.2165) | 5172 | 933 | 0.1804 | -0.041 | 0.0002 |
| 7:[0.2165,0.2677) | 5172 | 1226 | 0.237 | -0.385 | 0.0167 |
| 8:[0.2677,0.3526) | 5172 | 1562 | 0.302 | -0.716 | 0.0634 |
| 9:[0.3526,inf) | 5173 | 2665 | 0.5152 | -1.615 | 0.3818 |


### ENG:leverage_x_delinq  (IV=0.8286)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.0004544) | 5173 | 204 | 0.0394 | 1.637 | 0.1534 |
| 1:[0.0004544,0.0007415) | 5172 | 300 | 0.058 | 1.232 | 0.0996 |
| 2:[0.0007415,0.001049) | 5172 | 405 | 0.0783 | 0.911 | 0.0608 |
| 3:[0.001049,0.001402) | 5172 | 461 | 0.0891 | 0.769 | 0.0456 |
| 4:[0.001402,0.001826) | 5172 | 581 | 0.1123 | 0.513 | 0.0221 |
| 5:[0.001826,0.002342) | 5172 | 752 | 0.1454 | 0.217 | 0.0044 |
| 6:[0.002342,0.003033) | 5172 | 955 | 0.1846 | -0.069 | 0.0005 |
| 7:[0.003033,0.004078) | 5172 | 1156 | 0.2235 | -0.309 | 0.0105 |
| 8:[0.004078,0.006016) | 5172 | 1593 | 0.308 | -0.745 | 0.069 |
| 9:[0.006016,inf) | 5173 | 2617 | 0.5059 | -1.577 | 0.3629 |


### ENG:util_sq  (IV=0.7368)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.01983) | 5173 | 226 | 0.0437 | 1.53 | 0.1389 |
| 1:[0.01983,0.04724) | 5172 | 320 | 0.0619 | 1.164 | 0.0909 |
| 2:[0.04724,0.08292) | 5172 | 385 | 0.0744 | 0.965 | 0.067 |
| 3:[0.08292,0.1271) | 5172 | 471 | 0.0911 | 0.746 | 0.0432 |
| 4:[0.1271,0.1822) | 5172 | 641 | 0.1239 | 0.401 | 0.0141 |
| 5:[0.1822,0.2478) | 5172 | 773 | 0.1495 | 0.184 | 0.0032 |
| 6:[0.2478,0.3296) | 5172 | 956 | 0.1848 | -0.07 | 0.0005 |
| 7:[0.3296,0.4336) | 5172 | 1259 | 0.2434 | -0.42 | 0.0201 |
| 8:[0.4336,0.5774) | 5172 | 1553 | 0.3003 | -0.708 | 0.0618 |
| 9:[0.5774,inf) | 5173 | 2440 | 0.4717 | -1.44 | 0.2971 |


### aggregate_credit_utilization  (IV=0.7368)

_Aggregate credit utilization ratio from the credit bureau (0-1)._


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.1408) | 5173 | 226 | 0.0437 | 1.53 | 0.1389 |
| 1:[0.1408,0.2173) | 5172 | 320 | 0.0619 | 1.164 | 0.0909 |
| 2:[0.2173,0.288) | 5172 | 385 | 0.0744 | 0.965 | 0.067 |
| 3:[0.288,0.3565) | 5172 | 471 | 0.0911 | 0.746 | 0.0432 |
| 4:[0.3565,0.4268) | 5172 | 641 | 0.1239 | 0.401 | 0.0141 |
| 5:[0.4268,0.4978) | 5172 | 773 | 0.1495 | 0.184 | 0.0032 |
| 6:[0.4978,0.5741) | 5172 | 956 | 0.1848 | -0.07 | 0.0005 |
| 7:[0.5741,0.6585) | 5172 | 1259 | 0.2434 | -0.42 | 0.0201 |
| 8:[0.6585,0.7598) | 5172 | 1553 | 0.3003 | -0.708 | 0.0618 |
| 9:[0.7598,inf) | 5173 | 2440 | 0.4717 | -1.44 | 0.2971 |


### ENG:affordability_x_delinq  (IV=0.6508)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.002163) | 3475 | 132 | 0.038 | 1.674 | 0.1065 |
| 1:[0.002163,0.003691) | 3475 | 168 | 0.0483 | 1.423 | 0.0837 |
| 2:[0.003691,0.005384) | 3475 | 220 | 0.0633 | 1.138 | 0.059 |
| 3:[0.005384,0.007388) | 3475 | 285 | 0.082 | 0.86 | 0.0371 |
| 4:[0.007388,0.009815) | 3475 | 310 | 0.0892 | 0.768 | 0.0305 |
| 5:[0.009815,0.01306) | 3476 | 452 | 0.13 | 0.346 | 0.0072 |
| 6:[0.01306,0.0174) | 3474 | 619 | 0.1782 | -0.026 | 0.0 |
| 7:[0.0174,0.02395) | 3475 | 705 | 0.2029 | -0.186 | 0.0025 |
| 8:[0.02395,0.03683) | 3475 | 1056 | 0.3039 | -0.725 | 0.0438 |
| 9:[0.03683,inf) | 3476 | 1840 | 0.5293 | -1.671 | 0.2766 |
| MISSING | 16971 | 3237 | 0.1907 | -0.109 | 0.004 |


### observed_cash_balance_p10  (IV=0.5504)

_10th-percentile daily cash balance from the bank feed (dollars; can be negative). Null if no linked feed._


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,-2589) | 3475 | 1700 | 0.4892 | -1.511 | 0.2217 |
| 1:[-2589,-1301) | 3475 | 1041 | 0.2996 | -0.705 | 0.0411 |
| 2:[-1301,-394.4) | 3476 | 734 | 0.2112 | -0.236 | 0.004 |
| 3:[-394.4,408.7) | 3474 | 605 | 0.1742 | 0.002 | 0.0 |
| 4:[408.7,1146) | 3475 | 442 | 0.1272 | 0.371 | 0.0082 |
| 5:[1146,1892) | 3476 | 393 | 0.1131 | 0.505 | 0.0145 |
| 6:[1892,2689) | 3475 | 292 | 0.084 | 0.833 | 0.0351 |
| 7:[2689,3626) | 3474 | 251 | 0.0723 | 0.997 | 0.0475 |
| 8:[3626,4937) | 3475 | 192 | 0.0553 | 1.283 | 0.0713 |
| 9:[4937,inf) | 3476 | 137 | 0.0394 | 1.636 | 0.103 |
| MISSING | 16971 | 3237 | 0.1907 | -0.109 | 0.004 |


### ENG:cash_runway_days  (IV=0.5082)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,-0.5577) | 3475 | 1722 | 0.4955 | -1.536 | 0.2299 |
| 1:[-0.5577,-0.2421) | 3475 | 1048 | 0.3016 | -0.714 | 0.0423 |
| 2:[-0.2421,-0.06686) | 3476 | 720 | 0.2071 | -0.212 | 0.0032 |
| 3:[-0.06686,0.06382) | 3474 | 553 | 0.1592 | 0.11 | 0.0008 |
| 4:[0.06382,0.1802) | 3475 | 390 | 0.1122 | 0.513 | 0.0149 |
| 5:[0.1802,0.2999) | 3476 | 330 | 0.0949 | 0.7 | 0.0259 |
| 6:[0.2999,0.4525) | 3475 | 298 | 0.0858 | 0.811 | 0.0335 |
| 7:[0.4525,0.6608) | 3474 | 273 | 0.0786 | 0.906 | 0.0405 |
| 8:[0.6608,1.012) | 3475 | 234 | 0.0673 | 1.073 | 0.0536 |
| 9:[1.012,inf) | 3476 | 219 | 0.063 | 1.144 | 0.0594 |
| MISSING | 16971 | 3237 | 0.1907 | -0.109 | 0.004 |


### ENG:score_x_util  (IV=0.3741)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.1052) | 5173 | 274 | 0.053 | 1.328 | 0.112 |
| 1:[0.1052,0.1569) | 5172 | 413 | 0.0799 | 0.889 | 0.0584 |
| 2:[0.1569,0.2027) | 5172 | 526 | 0.1017 | 0.624 | 0.0315 |
| 3:[0.2027,0.2492) | 5172 | 733 | 0.1417 | 0.247 | 0.0056 |
| 4:[0.2492,0.2993) | 5172 | 853 | 0.1649 | 0.068 | 0.0004 |
| 5:[0.2993,0.3551) | 5172 | 918 | 0.1775 | -0.021 | 0.0 |
| 6:[0.3551,0.4201) | 5172 | 1033 | 0.1997 | -0.166 | 0.0029 |
| 7:[0.4201,0.4989) | 5172 | 1068 | 0.2065 | -0.208 | 0.0046 |
| 8:[0.4989,0.6073) | 5172 | 1322 | 0.2556 | -0.485 | 0.0273 |
| 9:[0.6073,inf) | 5173 | 1884 | 0.3642 | -0.997 | 0.1313 |


### requested_amount_to_observed_revenue  (IV=0.3584)

_Engineered ratio of requested amount to observed monthly revenue._


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.006306) | 5173 | 363 | 0.0702 | 1.029 | 0.0745 |
| 1:[0.006306,0.00792) | 5172 | 472 | 0.0913 | 0.744 | 0.0429 |
| 2:[0.00792,0.009326) | 5172 | 574 | 0.111 | 0.526 | 0.0232 |
| 3:[0.009326,0.01071) | 5172 | 653 | 0.1263 | 0.38 | 0.0127 |
| 4:[0.01071,0.0121) | 5172 | 734 | 0.1419 | 0.245 | 0.0055 |
| 5:[0.0121,0.01374) | 5172 | 807 | 0.156 | 0.134 | 0.0017 |
| 6:[0.01374,0.01573) | 5172 | 938 | 0.1814 | -0.047 | 0.0002 |
| 7:[0.01573,0.01843) | 5172 | 1182 | 0.2285 | -0.338 | 0.0127 |
| 8:[0.01843,0.0227) | 5172 | 1306 | 0.2525 | -0.469 | 0.0254 |
| 9:[0.0227,inf) | 5173 | 1995 | 0.3857 | -1.088 | 0.1595 |


### ENG:draw_to_daily_rev  (IV=0.2951)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.03825) | 3475 | 216 | 0.0622 | 1.158 | 0.0606 |
| 1:[0.03825,0.04817) | 3475 | 276 | 0.0794 | 0.895 | 0.0396 |
| 2:[0.04817,0.05694) | 3475 | 326 | 0.0938 | 0.713 | 0.0268 |
| 3:[0.05694,0.06542) | 3475 | 388 | 0.1117 | 0.519 | 0.0152 |
| 4:[0.06542,0.0742) | 3475 | 450 | 0.1295 | 0.351 | 0.0073 |
| 5:[0.0742,0.0841) | 3475 | 533 | 0.1534 | 0.154 | 0.0015 |
| 6:[0.0841,0.09647) | 3476 | 593 | 0.1706 | 0.027 | 0.0 |
| 7:[0.09647,0.1134) | 3474 | 750 | 0.2159 | -0.264 | 0.0051 |
| 8:[0.1134,0.1406) | 3475 | 874 | 0.2515 | -0.464 | 0.0167 |
| 9:[0.1406,inf) | 3476 | 1381 | 0.3973 | -1.137 | 0.1182 |
| MISSING | 16971 | 3237 | 0.1907 | -0.109 | 0.004 |


### ENG:loan_to_observed_annual_rev  (IV=0.2951)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,0.006028) | 3475 | 216 | 0.0622 | 1.158 | 0.0606 |
| 1:[0.006028,0.007591) | 3475 | 276 | 0.0794 | 0.895 | 0.0396 |
| 2:[0.007591,0.008974) | 3475 | 326 | 0.0938 | 0.713 | 0.0268 |
| 3:[0.008974,0.01031) | 3475 | 388 | 0.1117 | 0.519 | 0.0152 |
| 4:[0.01031,0.01169) | 3475 | 450 | 0.1295 | 0.351 | 0.0073 |
| 5:[0.01169,0.01325) | 3475 | 533 | 0.1534 | 0.154 | 0.0015 |
| 6:[0.01325,0.0152) | 3476 | 593 | 0.1706 | 0.027 | 0.0 |
| 7:[0.0152,0.01787) | 3474 | 750 | 0.2159 | -0.264 | 0.0051 |
| 8:[0.01787,0.02217) | 3475 | 874 | 0.2515 | -0.464 | 0.0167 |
| 9:[0.02217,inf) | 3476 | 1381 | 0.3973 | -1.137 | 0.1182 |
| MISSING | 16971 | 3237 | 0.1907 | -0.109 | 0.004 |


### ENG:draw_to_cash_buffer  (IV=0.285)

_engineered ratio_


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,2.788) | 13900 | 3855 | 0.2773 | -0.597 | 0.1145 |
| 1:[2.788,inf) | 20851 | 1932 | 0.0927 | 0.727 | 0.1665 |
| MISSING | 16971 | 3237 | 0.1907 | -0.109 | 0.004 |


### requested_amount  (IV=0.2791)

_Loan amount requested by the applicant in dollars (product range roughly $5K-$50K)._


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,1.619e+04) | 5173 | 382 | 0.0738 | 0.974 | 0.068 |
| 1:[1.619e+04,1.842e+04) | 5172 | 463 | 0.0895 | 0.765 | 0.0451 |
| 2:[1.842e+04,2.015e+04) | 5172 | 614 | 0.1187 | 0.45 | 0.0174 |
| 3:[2.015e+04,2.177e+04) | 5172 | 688 | 0.133 | 0.32 | 0.0092 |
| 4:[2.177e+04,2.337e+04) | 5172 | 804 | 0.1555 | 0.138 | 0.0018 |
| 5:[2.337e+04,2.501e+04) | 5172 | 876 | 0.1694 | 0.036 | 0.0001 |
| 6:[2.501e+04,2.68e+04) | 5172 | 960 | 0.1856 | -0.075 | 0.0006 |
| 7:[2.68e+04,2.888e+04) | 5172 | 1196 | 0.2312 | -0.353 | 0.0139 |
| 8:[2.888e+04,3.18e+04) | 5172 | 1302 | 0.2517 | -0.465 | 0.0249 |
| 9:[3.18e+04,inf) | 5173 | 1739 | 0.3362 | -0.874 | 0.098 |


### prior_approved_amount  (IV=0.2463)

_Amount the prior underwriter approved (may differ from requested). Null if declined._


| bin | n | bad | bad_rate | woe | iv |
| --- | --- | --- | --- | --- | --- |
| 0:[-inf,1.596e+04) | 5173 | 414 | 0.08 | 0.887 | 0.0581 |
| 1:[1.596e+04,1.814e+04) | 5172 | 499 | 0.0965 | 0.682 | 0.0369 |
| 2:[1.814e+04,1.986e+04) | 5172 | 599 | 0.1158 | 0.478 | 0.0195 |
| 3:[1.986e+04,2.147e+04) | 5172 | 705 | 0.1363 | 0.292 | 0.0077 |
| 4:[2.147e+04,2.304e+04) | 5172 | 813 | 0.1572 | 0.125 | 0.0015 |
| 5:[2.304e+04,2.468e+04) | 5172 | 913 | 0.1765 | -0.014 | 0.0 |
| 6:[2.468e+04,2.646e+04) | 5172 | 936 | 0.181 | -0.044 | 0.0002 |
| 7:[2.646e+04,2.853e+04) | 5172 | 1188 | 0.2297 | -0.344 | 0.0132 |
| 8:[2.853e+04,3.146e+04) | 5172 | 1263 | 0.2442 | -0.424 | 0.0205 |
| 9:[3.146e+04,inf) | 5173 | 1694 | 0.3275 | -0.834 | 0.0886 |


## 3. MNAR missingness (signal in the *fact* of missing)

| feature | pct_missing | dr_present | dr_missing | gap |
| --- | --- | --- | --- | --- |
| days_since_last_inquiry_elsewhere | 54.5 | 0.2139 | 0.1415 | -0.0725 |
| days_since_last_external_decline | 54.8 | 0.2128 | 0.1428 | -0.07 |
| observed_monthly_revenue_avg_3mo | 32.8 | 0.1665 | 0.1907 | 0.0242 |
| observed_revenue_trend_3mo | 32.8 | 0.1665 | 0.1907 | 0.0242 |
| observed_revenue_volatility | 32.8 | 0.1665 | 0.1907 | 0.0242 |
| observed_cash_balance_p10 | 32.8 | 0.1665 | 0.1907 | 0.0242 |
| observed_overdraft_count_3mo | 32.8 | 0.1665 | 0.1907 | 0.0242 |
| payroll_regularity_score | 32.8 | 0.1665 | 0.1907 | 0.0242 |


## 4. Redundancy (|Spearman| >= 0.70) - keep one per cluster

- `invoice_payment_delinquency_rate` <-> `ENG:leverage_x_delinq`  (+0.88)
- `invoice_payment_delinquency_rate` <-> `ENG:util_sq`  (+0.73)
- `invoice_payment_delinquency_rate` <-> `aggregate_credit_utilization`  (+0.73)
- `invoice_payment_delinquency_rate` <-> `ENG:affordability_x_delinq`  (+0.91)
- `invoice_payment_delinquency_rate` <-> `observed_cash_balance_p10`  (-0.97)
- `invoice_payment_delinquency_rate` <-> `ENG:cash_runway_days`  (-0.90)
- `ENG:leverage_x_delinq` <-> `ENG:util_sq`  (+0.78)
- `ENG:leverage_x_delinq` <-> `aggregate_credit_utilization`  (+0.78)
- `ENG:leverage_x_delinq` <-> `ENG:affordability_x_delinq`  (+0.98)
- `ENG:leverage_x_delinq` <-> `observed_cash_balance_p10`  (-0.85)
- `ENG:leverage_x_delinq` <-> `ENG:cash_runway_days`  (-0.72)
- `ENG:leverage_x_delinq` <-> `requested_amount_to_observed_revenue`  (+0.75)
- `ENG:leverage_x_delinq` <-> `ENG:draw_to_daily_rev`  (+0.78)
- `ENG:leverage_x_delinq` <-> `ENG:loan_to_observed_annual_rev`  (+0.78)
- `ENG:util_sq` <-> `aggregate_credit_utilization`  (+1.00)
- `ENG:util_sq` <-> `ENG:affordability_x_delinq`  (+0.80)
- `ENG:util_sq` <-> `ENG:score_x_util`  (+0.81)
- `aggregate_credit_utilization` <-> `ENG:affordability_x_delinq`  (+0.80)
- `aggregate_credit_utilization` <-> `ENG:score_x_util`  (+0.81)
- `ENG:affordability_x_delinq` <-> `observed_cash_balance_p10`  (-0.87)
- `ENG:affordability_x_delinq` <-> `ENG:cash_runway_days`  (-0.74)
- `ENG:affordability_x_delinq` <-> `requested_amount_to_observed_revenue`  (+0.79)
- `ENG:affordability_x_delinq` <-> `ENG:draw_to_daily_rev`  (+0.79)
- `ENG:affordability_x_delinq` <-> `ENG:loan_to_observed_annual_rev`  (+0.79)
- `observed_cash_balance_p10` <-> `ENG:cash_runway_days`  (+0.93)
- `requested_amount_to_observed_revenue` <-> `ENG:draw_to_daily_rev`  (+1.00)
- `requested_amount_to_observed_revenue` <-> `ENG:loan_to_observed_annual_rev`  (+1.00)
- `requested_amount_to_observed_revenue` <-> `ENG:req_to_stated_rev`  (+0.92)
- `requested_amount_to_observed_revenue` <-> `ENG:debt_service_ratio`  (+0.86)
- `ENG:draw_to_daily_rev` <-> `ENG:loan_to_observed_annual_rev`  (+1.00)
- `ENG:draw_to_daily_rev` <-> `ENG:req_to_stated_rev`  (+0.89)
- `ENG:draw_to_daily_rev` <-> `ENG:debt_service_ratio`  (+0.86)
- `ENG:loan_to_observed_annual_rev` <-> `ENG:req_to_stated_rev`  (+0.89)
- `ENG:loan_to_observed_annual_rev` <-> `ENG:debt_service_ratio`  (+0.86)
- `requested_amount` <-> `prior_approved_amount`  (+0.98)
- `ENG:req_to_stated_rev` <-> `ENG:debt_service_ratio`  (+0.83)
- `existing_debt_obligations` <-> `ENG:debt_service_ratio`  (+0.83)
- `observed_overdraft_count_3mo` <-> `ENG:overdraft_per_month`  (+1.00)