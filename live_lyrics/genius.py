"""Optional: find a song from a snippet of its lyrics via the Genius API (free token:
https://genius.com/api-clients -> "Generate Access Token")."""
import httpx


async def search(text: str, token: str, limit: int = 5) -> list[tuple[str, str]]:
    """[(title, artist), ...] for songs whose lyrics match `text`."""
    async with httpx.AsyncClient(timeout=10) as c:
        r = await c.get("https://api.genius.com/search", params={"q": text},
                        headers={"Authorization": f"Bearer {token}"})
        r.raise_for_status()
        hits = r.json()["response"]["hits"]
    return [(h["result"]["title"], h["result"]["primary_artist"]["name"])
            for h in hits if h.get("type") == "song"][:limit]
