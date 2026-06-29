"""
Booking.com Scraper – America's Cup Cagliari
Navigates search results, enters each property page and extracts reviews.
"""

import time
import random
import csv
import os
from datetime import datetime, date
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

SEARCH_URLS = {
    "cagliari": (
        "https://www.booking.com/searchresults.it.html?ss=Cagliari%2C+Sardegna%2C+Italia"
        "&lang=it&sb=1&dest_id=-112463&dest_type=city"
        "&group_adults=2&no_rooms=1&group_children=0"
    ),
}

OUTPUT_DIR  = "data/raw/"
SLEEP_MIN   = 2
SLEEP_MAX   = 5
MAX_PROPERTIES = 1000  # properties to visit per city
MAX_REVIEW_PAGES = 10000   # review pages per property

# ── Date filter ────────────────────────────────
# Edit these values to change the collection range.
# Format: "YYYY-MM-DD"  |  None = no limit
DATE_FROM = "2026-05-25"   # include reviews from this date
DATE_TO   = "2026-06-15"   # include reviews up to this date

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


# ──────────────────────────────────────────────
# DATE PARSING
# ──────────────────────────────────────────────

_IT_MONTHS = {
    "gennaio": "January", "febbraio": "February", "marzo": "March",
    "aprile": "April", "maggio": "May", "giugno": "June",
    "luglio": "July", "agosto": "August", "settembre": "September",
    "ottobre": "October", "novembre": "November", "dicembre": "December",
}

def parse_review_date(s: str):
    """Converts an Italian date string (e.g. '20 maggio 2026' or '1º maggio 2026') to a date object."""
    if not isinstance(s, str) or not s.strip():
        return None
    import re
    s = re.sub(r"(\d+)[ºª°]", r"\1", s)  # remove ordinal suffixes: 1º → 1
    for it, en in _IT_MONTHS.items():
        s = s.replace(it, en)
    try:
        return datetime.strptime(s.strip(), "%d %B %Y").date()
    except ValueError:
        return None

def _parse_limit(value):
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()

DATE_FROM_DT = _parse_limit(DATE_FROM)
DATE_TO_DT   = _parse_limit(DATE_TO)

def in_date_range(review_date) -> bool:
    """Returns True if the date falls within the configured range."""
    if review_date is None:
        return True  # if we can't parse the date, keep the review
    if DATE_FROM_DT and review_date < DATE_FROM_DT:
        return False
    if DATE_TO_DT and review_date > DATE_TO_DT:
        return False
    return True


# ──────────────────────────────────────────────
# BROWSER SETUP
# ──────────────────────────────────────────────

def get_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)

    ua = random.choice(USER_AGENTS)
    options.add_argument(f"user-agent={ua}")
    print(f"  🕵️  User-Agent: {ua[:60]}...")

    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    return driver


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────

def random_sleep():
    if random.random() < 0.15:
        t = random.uniform(15, 30)
        print(f"  ⏱ Long pause {t:.0f}s...")
    else:
        t = random.uniform(SLEEP_MIN, SLEEP_MAX)
        print(f"  ⏱ Waiting {t:.1f}s...")
    elapsed = 0
    while elapsed < t:
        chunk = random.uniform(0.3, 1.5)
        time.sleep(min(chunk, t - elapsed))
        elapsed += chunk


def scroll_page(driver, scrolls=3):
    last_height = driver.execute_script("return document.body.scrollHeight")
    for _ in range(scrolls):
        current = driver.execute_script("return window.pageYOffset")
        target = current + random.randint(400, 900)
        driver.execute_script(f"window.scrollTo(0, {target});")
        time.sleep(random.uniform(0.8, 2.0))
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(random.uniform(2, 4))
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height


def dismiss_cookie_banner(driver):
    """Closes the cookie banner if present."""
    selectors = [
        "button#onetrust-accept-btn-handler",
        "button[data-gdpr-consent='accept']",
        "button.cookie-consent__accept",
        "#cookie_action_close_header",
    ]
    for sel in selectors:
        try:
            btn = WebDriverWait(driver, 4).until(EC.element_to_be_clickable((By.CSS_SELECTOR, sel)))
            driver.execute_script("arguments[0].click();", btn)
            print("  🍪 Cookie banner dismissed.")
            time.sleep(1)
            return
        except TimeoutException:
            continue


def save_to_csv(data: list, city: str, source: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    date_range = f"{DATE_FROM or 'start'}_to_{DATE_TO or 'end'}"
    filename = f"{OUTPUT_DIR}{source}_{city}_{date_range}.csv"

    if not data:
        print("  ⚠️  No data to save.")
        return

    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

    print(f"  ✅ Saved {len(data)} reviews to: {filename}")


def checkpoint_path(city: str) -> str:
    return f"{OUTPUT_DIR}checkpoint_{city}.txt"

def load_checkpoint(city: str) -> set:
    path = checkpoint_path(city)
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(line.strip() for line in f if line.strip())

def save_checkpoint(city: str, url: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(checkpoint_path(city), "a", encoding="utf-8") as f:
        f.write(url + "\n")

def clear_checkpoint(city: str):
    path = checkpoint_path(city)
    if os.path.exists(path):
        os.remove(path)


# ──────────────────────────────────────────────
# STEP 1 – COLLECT PROPERTY LINKS
# ──────────────────────────────────────────────

def _collect_links_from_page(driver, links: list, max_properties: int) -> int:
    """Collects links from property cards visible on the current page; returns the number of new links found."""
    soup = BeautifulSoup(driver.page_source, "html.parser")
    found = 0
    cards = soup.find_all("div", {"data-testid": "property-card"})
    for card in cards:
        a = card.find("a", {"data-testid": "title-link"})
        if a and a.get("href"):
            href = a["href"]
            if not href.startswith("http"):
                href = "https://www.booking.com" + href
            if href not in links:
                links.append(href)
                found += 1
        if len(links) >= max_properties:
            break

    # Fallback
    if found == 0 and not cards:
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/hotel/" in href and href not in links:
                if not href.startswith("http"):
                    href = "https://www.booking.com" + href
                links.append(href)
                found += 1
            if len(links) >= max_properties:
                break

    return found


def get_property_links(driver, search_url: str, max_properties: int) -> list:
    """
    Opens the Booking search results page and collects links to individual
    properties by clicking 'Load more results'.
    """
    print(f"\n🔎 Loading search results...")
    driver.get(search_url)
    time.sleep(random.uniform(5, 10))
    dismiss_cookie_banner(driver)
    scroll_page(driver, scrolls=4)

    links = []
    page_num = 1

    while len(links) < max_properties:
        found = _collect_links_from_page(driver, links, max_properties)
        print(f"  → page {page_num}: {found} new properties (total: {len(links)})")

        if len(links) >= max_properties:
            break

        # Look for the "Load more results" button by text
        try:
            load_more = WebDriverWait(driver, 8).until(
                EC.element_to_be_clickable((By.XPATH, "//button[normalize-space()='Carica più risultati']"))
            )
            driver.execute_script("arguments[0].scrollIntoView(true);", load_more)
            time.sleep(1)
            driver.execute_script("arguments[0].click();", load_more)
            print(f"  🔄 'Load more results' clicked")
            time.sleep(random.uniform(4, 7))
            scroll_page(driver, scrolls=3)
            page_num += 1
        except TimeoutException:
            print("  ⏹ 'Load more results' button not found, end of results.")
            break

    print(f"\n  ✅ Total properties collected: {len(links)}")
    return links


# ──────────────────────────────────────────────
# STEP 2 – EXTRACT REVIEWS FROM A PROPERTY
# ──────────────────────────────────────────────

def scrape_property_reviews(driver, property_url: str, city: str, max_pages: int) -> list:
    """Visits the property page, clicks on the reviews tab and extracts review data."""
    # Property name from URL
    try:
        property_name = property_url.split("/hotel/it/")[1].split(".it.html")[0].replace("-", " ").title()
    except (IndexError, AttributeError):
        property_name = "unknown"

    print(f"\n  🏨 Property: {property_name}")
    driver.get(property_url)
    time.sleep(random.uniform(5, 10))
    dismiss_cookie_banner(driver)
    scroll_page(driver, scrolls=2)

    # Property stars/category (extracted once from the property page)
    property_stars = ""
    try:
        prop_soup = BeautifulSoup(driver.page_source, "html.parser")
        stars_el = (
            prop_soup.find(attrs={"data-testid": "rating-squares"}) or
            prop_soup.find(attrs={"data-testid": "quality-rating"})
        )
        if stars_el:
            import re as _re
            aria = stars_el.get("aria-label", "")
            m = _re.search(r"\d+", aria)
            if m:
                property_stars = int(m.group())
            else:
                icons = stars_el.find_all("span", attrs={"aria-hidden": "true"})
                property_stars = len(icons) if icons else ""
    except Exception:
        pass
    print(f"    ⭐ Property stars: {property_stars or 'N/A'}")

    # Click the "Reviews" tab in the property navigation
    try:
        tab = WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "[data-testid='Property-Header-Nav-Tab-Trigger-reviews']"))
        )
        driver.execute_script("arguments[0].click();", tab)
        print("    ✅ Reviews tab clicked")
        time.sleep(random.uniform(3, 5))
    except TimeoutException:
        print("    ⚠️  Reviews tab not found, continuing on current page")

    all_reviews = []

    for page_num in range(max_pages):
        print(f"    📄 Review page {page_num + 1}/{max_pages}")
        print(f"    🌐 URL: {driver.current_url[:100]}")
        scroll_page(driver, scrolls=2)
        soup = BeautifulSoup(driver.page_source, "html.parser")

        # Debug: show page title
        title_tag = soup.find("title")
        print(f"    📋 Page title: {title_tag.get_text(strip=True)[:80] if title_tag else 'N/A'}")

        review_blocks = (
            soup.find_all(attrs={"data-testid": "review-card"}) or
            soup.find_all(attrs={"data-testid": "featuredreview"}) or
            soup.find_all(attrs={"data-testid": "review-block"}) or
            soup.find_all("div", class_=lambda c: c and "review_item" in c) or
            soup.find_all("div", {"itemprop": "review"})
        )

        if not review_blocks:
            debug_path = f"data/raw/debug_booking_{property_name[:20]}_{page_num}.html"
            os.makedirs("data/raw", exist_ok=True)
            with open(debug_path, "w", encoding="utf-8") as f:
                f.write(driver.page_source)
            print(f"    🛠 No review blocks found — HTML saved to: {debug_path}")

        print(f"      → {len(review_blocks)} reviews found")

        stop_early = False
        for block in review_blocks:
            review = extract_booking_review(block, city, property_name, property_stars)
            if not review:
                continue
            rd = parse_review_date(review["review_date"])
            if DATE_FROM_DT and rd and rd < DATE_FROM_DT:
                # Booking sorts from most recent: everything below is out of range
                stop_early = True
                break
            if in_date_range(rd):
                all_reviews.append(review)

        if stop_early:
            print(f"    ⏹ Reviews older than {DATE_FROM} — pagination stopped.")
            break

        # Next page
        try:
            next_btn = driver.find_element(By.CSS_SELECTOR, "button[aria-label='pagina successiva']")
            if not next_btn.is_enabled():
                break
            next_btn.click()
            random_sleep()
        except NoSuchElementException:
            try:
                next_btn = driver.find_element(By.CSS_SELECTOR, "a.pagenext")
                next_btn.click()
                random_sleep()
            except NoSuchElementException:
                print("    ⏹ End of pagination.")
                break

    return all_reviews


def extract_booking_review(block, city: str, property_name: str, property_stars="") -> dict:
    """Extracts fields from a single Booking review block."""
    try:
        pos_el = (
            block.find(attrs={"data-testid": "review-positive-text"}) or
            block.find(attrs={"data-testid": "featuredreview-text"}) or
            block.find(attrs={"data-testid": "featuredreviewcard-text"})
        )
        text_positive = pos_el.get_text(strip=True).strip("«»").strip() if pos_el else ""

        neg_el = block.find(attrs={"data-testid": "review-negative-text"})
        text_negative = neg_el.get_text(strip=True).strip("«»").strip() if neg_el else ""

        text = text_positive  # kept for backward compatibility with the `if not text` check below

        # Review title
        title_el = (
            block.find(attrs={"data-testid": "review-title"}) or
            block.find(attrs={"data-testid": "featuredreview-title"})
        )
        review_title = title_el.get_text(strip=True).strip("«»").strip() if title_el else ""

        # Numeric rating (e.g. "8,5" or "8.5")
        rating = ""
        score_el = (
            block.find(attrs={"data-testid": "review-score"}) or
            block.find(attrs={"data-testid": "review-score-badge"}) or
            block.find(class_=lambda c: c and "review-score" in c if c else False)
        )
        if score_el:
            import re
            raw = score_el.get_text(separator=" ", strip=True).replace(",", ".")
            m = re.search(r"\d+\.\d+|\d+", raw)
            if m:
                val = float(m.group())
                rating = val if val <= 10 else round(val / 10, 2)

        # Reviewer name
        avatar_el = block.find(attrs={"data-testid": "review-avatar"}) or \
                    block.find(attrs={"data-testid": "featuredreview-avatar"})
        reviewer = ""
        if avatar_el:
            name_div = avatar_el.find("div", class_=lambda c: c and "b08850ce41" in c if c else False)
            reviewer = name_div.get_text(strip=True) if name_div else ""

        # Reviewer country
        country_el = block.find("img", alt=True, src=lambda s: s and "images-flags" in s if s else False)
        country = country_el["alt"] if country_el else ""

        # Stay type (e.g. "Coppia", "Famiglia", "Solo", "Business")
        stay_type = ""
        stay_el = (
            block.find(attrs={"data-testid": "review-traveler-type"}) or
            block.find(attrs={"data-testid": "review-stay-type"})
        )
        if stay_el:
            stay_type = stay_el.get_text(strip=True)
        else:
            # Fallback: look for text matching typical Booking stay-type keywords
            for el in block.find_all(["span", "div", "p"]):
                txt = el.get_text(strip=True)
                if txt in ("Coppia", "Solo", "Famiglia con bambini piccoli",
                           "Famiglia con bambini più grandi", "Gruppo", "Viaggio d'affari"):
                    stay_type = txt
                    break

        # Number of nights
        nights_stayed = ""
        import re as _re
        nights_el = block.find(attrs={"data-testid": "review-num-nights"})
        if nights_el:
            txt = nights_el.get_text(strip=True)
            m = _re.search(r"\d+", txt)
            nights_stayed = int(m.group()) if m else ""
        else:
            for el in block.find_all(["span", "div", "p"]):
                txt = el.get_text(strip=True)
                m = _re.match(r"^(\d+)\s*nott", txt, _re.IGNORECASE)
                if m:
                    nights_stayed = int(m.group(1))
                    break

        # Review date: review-card exposes "Data della recensione: 20 maggio 2026"
        date_raw = ""
        date_el = block.find(attrs={"data-testid": "review-date"})
        if date_el:
            date_raw = date_el.get_text(strip=True).replace("Data della recensione:", "").strip()
        if not date_raw:
            stay_el = block.find(attrs={"data-testid": "review-stay-date"})
            if stay_el:
                date_raw = stay_el.get_text(strip=True)

        if not text:
            return None

        return {
            "city":             city,
            "source":           "booking",
            "property":         property_name,
            "property_stars":   property_stars,
            "scraped_at":       datetime.now().isoformat(),
            "review_date":      date_raw,
            "rating":           rating,
            "review_title":     review_title,
            "stay_type":        stay_type,
            "nights_stayed":    nights_stayed,
            "text_positive":    text_positive,
            "text_negative":    text_negative,
            "reviewer":         reviewer,
            "reviewer_country": country,
        }

    except Exception as e:
        print(f"      ⚠️  Error parsing review: {e}")
        return None


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print(f"📅 Date range: {DATE_FROM or '∞'} → {DATE_TO or '∞'}")
    for city, search_url in SEARCH_URLS.items():
        done = load_checkpoint(city)
        if done:
            print(f"♻️  Checkpoint found: {len(done)} properties already processed, resuming.")

        driver = get_driver()
        all_reviews = []

        # Load already saved reviews if resuming from checkpoint
        date_range = f"{DATE_FROM or 'start'}_to_{DATE_TO or 'end'}"
        csv_file = f"{OUTPUT_DIR}booking_{city}_{date_range}.csv"
        if done and os.path.exists(csv_file):
            import csv as _csv
            with open(csv_file, "r", encoding="utf-8") as f:
                all_reviews = list(_csv.DictReader(f))
            print(f"  📂 Loaded {len(all_reviews)} previously saved reviews.")

        try:
            property_links = get_property_links(driver, search_url, MAX_PROPERTIES)
            to_process = [u for u in property_links if u not in done]
            skipped = len(property_links) - len(to_process)
            if skipped:
                print(f"  ⏭ Skipped {skipped} already processed properties.")

            for i, prop_url in enumerate(to_process):
                print(f"\n[{i+1}/{len(to_process)}] {prop_url[:80]}...")
                try:
                    reviews = scrape_property_reviews(driver, prop_url, city, MAX_REVIEW_PAGES)
                    all_reviews.extend(reviews)
                    save_to_csv(all_reviews, city=city, source="booking")
                except Exception as e:
                    print(f"  ⚠️  Property skipped due to error: {e}")
                    # If the driver crashed, create a new one
                    try:
                        driver.quit()
                    except Exception:
                        pass
                    driver = get_driver()
                finally:
                    save_checkpoint(city, prop_url)
                random_sleep()

        except Exception as e:
            print(f"❌ General error: {e}")
        finally:
            driver.quit()

        save_to_csv(all_reviews, city=city, source="booking")
        clear_checkpoint(city)
        print(f"\n📊 Total reviews collected for {city}: {len(all_reviews)}")

    print("\n🏁 Scraping complete.")
