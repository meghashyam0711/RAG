# Acme Gateway API Specification v3.2

<!-- page: 1 -->

## 1. Overview

The Acme Gateway fronts all public traffic for the Acme platform. Clients authenticate with an API key passed in the `X-Acme-Key` header. All endpoints are versioned under `/v3`.

The gateway is deployed in three regions (us-east, eu-west, ap-south). Requests are routed to the nearest healthy region.

<!-- page: 4 -->

## 2. Rate Limiting

Rate limiting uses a token bucket per API key. The parameter `max_requests_per_minute` is defined as the sustained number of requests a single key may issue per minute and defaults to 600. The parameter `burst_size` defaults to 100 and controls how many requests may be issued instantaneously above the sustained rate.

When a client exceeds its limit the gateway returns HTTP 429 with a `Retry-After` header. Clients must not retry before the `Retry-After` interval elapses.

Enterprise-tier limits are not defined here; see Appendix B for the overrides that apply to Enterprise keys.

### 2.1 Default Limits by Endpoint

| Endpoint | max_requests_per_minute | burst_size | timeout_ms |
|---|---|---|---|
| GET /v3/orders | 600 | 100 | 3000 |
| POST /v3/orders | 120 | 20 | 5000 |
| GET /v3/reports | 60 | 10 | 30000 |
| DELETE /v3/orders/{id} | 30 | 5 | 5000 |

<!-- page: 7 -->

## 3. Authentication

API keys are scoped to a single project. Keys rotate every 90 days. The `key_rotation_days` parameter cannot be configured by clients. Webhook signatures use HMAC-SHA256 and are not encrypted.

## 4. Error Handling

Errors use RFC 9457 problem details. The gateway does not retry failed upstream calls for non-idempotent methods (POST, PATCH). Idempotent GET requests are retried at most 2 times with exponential backoff, as specified by `upstream_retry_limit`.

<!-- page: 41 -->

## 7. Operational Metrics (in millions)

Quarterly gateway request volume and cost, $ in millions except request counts which are in millions of requests.

| Metric | Q3 FY2024 | Q3 FY2025 |
|---|---|---|
| Requests served | 1,840 | 2,310 |
| Infrastructure cost | 4.2 | 5.1 |
| Net savings from caching | 0.9 | (0.3) |

The negative net savings in Q3 FY2025, shown as (0.3), reflect the cache cluster migration and did not recur in the following quarter.

<!-- page: 123 -->

## Appendix B: Enterprise Tier Overrides

This appendix supersedes the default limits in Section 2 for keys on the Enterprise tier. For Enterprise keys `max_requests_per_minute` is set to 6000 and `burst_size` is set to 1000. The `timeout_ms` values in Section 2.1 are unchanged.

Enterprise overrides do not apply to DELETE endpoints, which keep the default limits.
