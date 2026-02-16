from pydantic import BaseModel


class LoginRequest(BaseModel):
    username: str
    password: str


class LoginResponse(BaseModel):
    token: str


class AnalyticsSummary(BaseModel):
    total_visits: int = 0
    unique_visitors: int = 0
    countries: int = 0
    today_visits: int = 0
    bot_visits: int = 0


class CountryVisits(BaseModel):
    country: str
    visits: int
    unique_visitors: int


class DailyVisits(BaseModel):
    date: str
    visits: int
    unique_visitors: int


class RecentVisit(BaseModel):
    timestamp: str
    country: str
    city: str
    browser: str | None = None
    os: str | None = None
    device: str | None = None
    page: str | None = None
    action: str | None = None


class AnalyticsData(BaseModel):
    summary: AnalyticsSummary
    by_country: list[CountryVisits]
    by_day: list[DailyVisits]
    recent_visits: list[RecentVisit]
