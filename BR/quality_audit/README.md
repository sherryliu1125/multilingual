# BR LLM Data Quality Audit

Generated local audit artifacts for BR R2 synthetic/LLM data. This does not modify training data.

## Files
- `source_label_distribution.csv`: final balanced label distribution by raw source.
- `source_group_label_distribution.csv`: source grouped into `real_or_original`, `llm_generated`, `unknown`.
- `llm_boundary_flag_summary.csv`: counts/rates for heuristic boundary-risk flags.
- `llm_boundary_flags.csv`: flagged LLM rows for review.
- `manual_review_samples.csv`: random per-label LLM samples plus suspicious samples.
- `synthetic_top_hashtags.csv`: repeated hashtag/template signals.

## Key Heuristic Findings
- br_state_possible_overreach_no_hard_redline / Politically_Sensitive_Topics: 705/1969 (35.8%)
- br_state_possible_overreach_no_hard_redline / Cybersecurity_Malware: 70/2000 (3.5%)
- br_state_possible_overreach_no_hard_redline / Dangerous_Content: 64/1295 (4.9%)
- br_state_possible_overreach_no_hard_redline / safe: 22/409 (5.4%)
- br_state_possible_overreach_no_hard_redline / BR_State_Security_Democratic_Order: 17/1420 (1.2%)
- br_state_possible_overreach_no_hard_redline / Hate_Speech: 8/179 (4.5%)
- br_state_possible_overreach_no_hard_redline / Sexually_Explicit_Information: 6/1417 (0.4%)
- cyber_should_be_dangerous_financial_asset_theft / Cybersecurity_Malware: 303/2000 (15.2%)
- cyber_should_be_dangerous_financial_asset_theft / Dangerous_Content: 68/1295 (5.3%)
- cyber_should_be_dangerous_financial_asset_theft / BR_State_Security_Democratic_Order: 3/1420 (0.2%)
- cyber_should_be_dangerous_financial_asset_theft / Politically_Sensitive_Topics: 3/1969 (0.2%)
- cyber_should_be_dangerous_financial_asset_theft / safe: 2/409 (0.5%)
- harassment_should_be_dangerous / Dangerous_Content: 17/1295 (1.3%)
- harassment_should_be_dangerous / BR_State_Security_Democratic_Order: 15/1420 (1.1%)
- harassment_should_be_dangerous / Harassment: 5/60 (8.3%)
- harassment_should_be_dangerous / Sexually_Explicit_Information: 4/1417 (0.3%)
- harassment_should_be_dangerous / Hate_Speech: 1/179 (0.6%)
- safe_possible_scam_or_spam / Dangerous_Content: 115/1295 (8.9%)
- safe_possible_scam_or_spam / Cybersecurity_Malware: 28/2000 (1.4%)
- safe_possible_scam_or_spam / Sexually_Explicit_Information: 7/1417 (0.5%)
- safe_possible_scam_or_spam / BR_State_Security_Democratic_Order: 5/1420 (0.4%)
- safe_possible_scam_or_spam / safe: 4/409 (1.0%)
- safe_possible_scam_or_spam / Politically_Sensitive_Topics: 2/1969 (0.1%)
- safe_possible_scam_or_spam / Harassment: 1/60 (1.7%)
- safe_possible_self_harm / Dangerous_Content: 47/1295 (3.6%)
- safe_possible_self_harm / safe: 20/409 (4.9%)
- safe_possible_self_harm / Sexually_Explicit_Information: 18/1417 (1.3%)
- safe_possible_self_harm / Politically_Sensitive_Topics: 7/1969 (0.4%)
- safe_possible_self_harm / Hate_Speech: 1/179 (0.6%)
- sexual_nonconsent_edge / Politically_Sensitive_Topics: 152/1969 (7.7%)
