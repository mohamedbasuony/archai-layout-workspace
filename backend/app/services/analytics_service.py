"""SQLite analytics queries."""

import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

from app.config import settings


def get_analytics_data(days_filter: int = 30) -> dict:
    """Fetch analytics data from the SQLite database.

    Returns a dict with keys: summary, by_country, by_day, recent_visits.
    """
    try:
        conn = sqlite3.connect(settings.analytics_db_path)
        conn.row_factory = sqlite3.Row

        if days_filter > 0:
            date_cutoff = (datetime.now() - timedelta(days=days_filter)).strftime("%Y-%m-%d")
            date_filter = f"WHERE DATE(timestamp) >= '{date_cutoff}'"
        else:
            date_filter = ""

        # By country
        country_rows = conn.execute(
            f"""
            SELECT
                COALESCE(NULLIF(country, ''), 'Unknown') as country,
                COUNT(*) as visits,
                COUNT(DISTINCT visitor_id) as unique_visitors
            FROM visits
            {date_filter}
            GROUP BY country
            ORDER BY visits DESC
            """
        ).fetchall()
        by_country = [dict(r) for r in country_rows]

        # By day
        daily_rows = conn.execute(
            f"""
            SELECT
                DATE(timestamp) as date,
                COUNT(*) as visits,
                COUNT(DISTINCT visitor_id) as unique_visitors
            FROM visits
            {date_filter}
            GROUP BY DATE(timestamp)
            ORDER BY date DESC
            LIMIT 30
            """
        ).fetchall()
        by_day = [dict(r) for r in daily_rows]

        # Recent visits
        recent_rows = conn.execute(
            f"""
            SELECT
                timestamp,
                COALESCE(NULLIF(country, ''), 'Unknown') as country,
                COALESCE(NULLIF(city, ''), 'Unknown') as city,
                browser, os, device, page, action
            FROM visits
            {date_filter}
            ORDER BY timestamp DESC
            LIMIT 100
            """
        ).fetchall()
        recent_visits = [dict(r) for r in recent_rows]

        # Summary
        today = datetime.now().strftime("%Y-%m-%d")
        summary_row = conn.execute(
            f"""
            SELECT
                COUNT(*) as total_visits,
                COUNT(DISTINCT visitor_id) as unique_visitors,
                COUNT(DISTINCT COALESCE(NULLIF(country, ''), 'Unknown')) as countries,
                SUM(CASE WHEN DATE(timestamp) = '{today}' THEN 1 ELSE 0 END) as today_visits,
                SUM(CASE WHEN is_bot = 1 THEN 1 ELSE 0 END) as bot_visits
            FROM visits
            {date_filter}
            """
        ).fetchone()
        summary = dict(summary_row) if summary_row else {}

        conn.close()
        return {
            "summary": summary,
            "by_country": by_country,
            "by_day": by_day,
            "recent_visits": recent_visits,
        }

    except Exception as e:
        print(f"Analytics DB error: {e}")
        return {
            "summary": {},
            "by_country": [],
            "by_day": [],
            "recent_visits": [],
        }
