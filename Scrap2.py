import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# 1. Setup undetected-chromedriver dengan stealth options
options = uc.ChromeOptions()
options.add_argument("--disable-notifications")
options.add_argument("--start-maximized")
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                     "AppleWebKit/537.36 (KHTML, like Gecko) "
                     "Chrome/113.0.0.0 Safari/537.36")
# Non-aktifkan extension automation
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

driver = uc.Chrome(options=options)

def scrape_reviews(name, url, max_scroll=50, scroll_pause=2):
    driver.get(url)
    time.sleep(5)  # tunggu page dan script-nya load

    # 1. Klik tombol "Semua ulasan" / "ulasan"
    try:
        btn = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, "//button[contains(@aria-label, 'ulasan') or contains(text(), 'ulasan')]")
            )
        )
        btn.click()
        time.sleep(5)
    except:
        print(f"[!] Tombol ulasan tidak ditemukan pada {name}, lanjut.")

    # 2. Scroll hingga load semua review
    feed = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, "//div[@role='feed']"))
    )
    prev = 0
    for i in range(max_scroll):
        driver.execute_script("arguments[0].scrollTop = arguments[0].scrollHeight", feed)
        time.sleep(scroll_pause)
        cards = feed.find_elements(By.XPATH, ".//div[@data-review-id]")
        if len(cards) == prev:
            break
        prev = len(cards)

    # 3. Ekstrak data tiap review
    data = []
    for card in cards:
        try:
            author    = card.find_element(By.XPATH, ".//span[contains(@class,'TSUbDb')]").text
            rating    = card.find_element(By.XPATH,
                            ".//span[contains(@aria-label,'bintang')]"
                        ).get_attribute("aria-label")
            text      = card.find_element(By.XPATH, ".//span[@jsname='bN97Pc']").text
            timestamp = card.find_element(By.XPATH, ".//span[contains(@class,'rsqaWe')]").text
            data.append({
                "Location": name,
                "Author": author,
                "Rating": rating,
                "Review Text": text,
                "Time": timestamp
            })
        except:
            continue

    print(f"[✓] {len(data)} ulasan diambil dari {name}")
    return data

# Contoh penggunaan
locations = [
    {
        "name":"Bank Jateng KCP Nguter",
        "url":"https://www.google.com/maps/place/Bank+Jateng+Cabang+Pembantu+Nguter+Sukoharjo/@-7.6854661,110.8555202,13z/data=!4m8!3m7!1s0x2e7a3ae8fef115db:0x45fb06b00fbcd3f9!8m2!3d-7.7357366!4d110.8725387!9m1!1b1!16s%2Fg%2F1hm49bk0g?entry=ttu&g_ep=EgoyMDI1MDQyMy4wIKXMDSoJLDEwMjExNDU1SAFQAw%3D%3D"
    },
    # … tambahkan lokasi lain di list ini
]

all_reviews = []
for loc in locations:
    all_reviews += scrape_reviews(loc["name"], loc["url"])

# Simpan ke CSV
df = pd.DataFrame(all_reviews)
df.to_csv("bank_Jateng_reviews.csv", index=False, encoding="utf-8")
print("Selesai: semua ulasan tersimpan di bank_Jateng_reviews.csv")

driver.quit()