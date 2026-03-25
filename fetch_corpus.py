import os
import re
import requests
import io
import PyPDF2
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from langdetect import detect, LangDetectException

# Disable SSL warnings
requests.packages.urllib3.disable_warnings()

exact_urls = [
    "https://www.iitj.ac.in/",
    "https://iitj.ac.in/office-of-academics/en/academic-regulations",
    "https://www.iitj.ac.in/PageImages/Gallery/06-2025/Ph.D._New.pdf",
    "https://www.iitj.ac.in/computer-science-engineering/en/Research-Archive",
    "https://www.iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
    "https://www.iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://www.iitj.ac.in/PageImages/Gallery/03-2025/Website-Research-Projects-638772906605230764.pdf",
    "https://iitj.ac.in/office-of-academics/en/list-of-academic-programs",
    "https://www.iitj.ac.in/bachelor-of-technology/en/hostels-facilities",
    "https://www.iitj.ac.in/Bachelor-of-Technology/en/Bachelor-of-Technology",
    "https://www.iitj.ac.in/mechanical-engineering/en/undergraduate-program",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/undergraduate-program",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/en/btech",
    "https://www.iitj.ac.in/metallurgical-and-materials-engineering/en/undergraduate-program",
    "https://www.iitj.ac.in/chemical-engineering/en/undergraduate-program",
    "https://www.iitj.ac.in/computer-science-engineering/en/undergraduate-programs",
    "https://www.iitj.ac.in/electrical-engineering/en/undergraduate-program",
    "https://www.iitj.ac.in/bioscience-bioengineering/en/undergraduate-program",
    "https://www.iitj.ac.in/PageImages/Gallery/07-2025/Academic-Calendar-AY-202526SemI2-with-CCCD-events-638871414539740843.pdf",
    "https://www.iitj.ac.in/computer-science-engineering",
    "https://www.iitj.ac.in/bioscience-bioengineering",
    "https://www.iitj.ac.in/health-center/en/health-center",
    "https://www.iitj.ac.in/health-center/en/services"
]

sublink_target_urls = [
    "https://www.iitj.ac.in/m/Index/main-departments?lg=en",
    "https://www.iitj.ac.in/faculty-positions/en/faculty-positions",
    "https://iitj.ac.in/office-of-academics/en/curriculum",
    "https://www.iitj.ac.in/faculty-positions/en/faculty-positions",
    "https://spc.iitj.ac.in/",
    "https://www.iitj.ac.in/schools/en/School-of-Management-&-Entrepreneurship",
    "https://www.iitj.ac.in/main/en/news",
    "https://libraryopac.iitj.ac.in/cgi-bin/koha/opac-search.pl?idx=&q=&idx=kw&sort_by=acqdate_dsc&do=OK&limit=mc-itype%3AEBK&weight_search=1"
]

urls_to_visit = set(exact_urls + sublink_target_urls)
visited = set()
collected_paragraphs = []

headers = {'User-Agent': 'Mozilla/5.0'}

def get_sublinks(url, soup):
    links = []
    base_domain = urlparse(url).netloc
    for a in soup.find_all('a', href=True):
        href = a['href']
        full_url = urljoin(url, href)
        if urlparse(full_url).netloc == base_domain and "#" not in href:
            if not full_url.endswith('.png') and not full_url.endswith('.jpg'):
                links.append(full_url)
    return links

def extract_pdf_text(content):
    text = ""
    try:
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n\n"
    except Exception as e:
        print("PDF Error:", e)
    return text

def extract_html_text(soup):
    for script in soup(["script", "style", "nav", "footer", "header"]):
        script.extract()
    return soup.get_text(separator='\n')

def is_english(text):
    if len(text.strip()) < 10: return False # Skip too short
    try:
        if detect(text) == 'en': return True
    except LangDetectException:
        pass
    return False
    
def clean_text(text):
    # Boilerplate check: skip lines that look like generic navigations
    boilerplate_terms = ['home', 'contact us', 'about us', 'sitemap', 'copyright', 'all rights reserved', 'login', 'read more']
    
    cleaned_docs = []
    paragraphs = text.split('\n')
    for p in paragraphs:
        p_lower = p.lower().strip()
        if len(p_lower) < 15: continue
        if any(p_lower == term for term in boilerplate_terms): continue
        
        # Remove text from other languages
        if not is_english(p_lower): continue
        
        # Lower-casing
        p_clean = p.lower()
        # Removal of excessive punctuation and non-textual content
        p_clean = re.sub(r'[^a-z0-9\s]', ' ', p_clean)
        # Tokenization & removing extra spaces
        tokens = p_clean.split()
        if len(tokens) > 3:
            cleaned_docs.append(tokens)
            
    return cleaned_docs

def scrape_urls():
    global urls_to_visit
    
    # Pre-collect sublinks
    for root_url in sublink_target_urls:
        try:
            print(f"Collecting sublinks for {root_url}")
            resp = requests.get(root_url, headers=headers, verify=False, timeout=10)
            if resp.status_code == 200 and 'text/html' in resp.headers.get('Content-Type', ''):
                soup = BeautifulSoup(resp.content, 'html.parser')
                sublinks = get_sublinks(root_url, soup)
                for sl in sublinks[:30]: # Limit to 30 sublinks per page to avoid infinite crawl
                    urls_to_visit.add(sl)
        except Exception as e:
            print(f"Failed to get sublinks for {root_url}: {e}")

    urls_to_process = list(urls_to_visit)
    print(f"Total URLs to process: {len(urls_to_process)}")
    
    all_cleaned_documents = []
    
    for url in urls_to_process:
        if url in visited: continue
        visited.add(url)
        print(f"Fetching: {url}")
        
        try:
            resp = requests.get(url, headers=headers, verify=False, timeout=10)
            if resp.status_code != 200: continue
            
            content_type = resp.headers.get('Content-Type', '')
            text = ""
            if 'application/pdf' in content_type or url.endswith('.pdf'):
                text = extract_pdf_text(resp.content)
            elif 'text/html' in content_type:
                soup = BeautifulSoup(resp.content, 'html.parser')
                text = extract_html_text(soup)
            else:
                text = extract_pdf_text(resp.content) if resp.content.startswith(b'%PDF') else resp.text
                
            cleaned_docs = clean_text(text)
            all_cleaned_documents.extend(cleaned_docs)
            
        except Exception as e:
            print(f"Error on {url}: {e}")

    print(f"Extraction complete. Found {len(all_cleaned_documents)} clean paragraphs.")
    
    # Save the cleaned corpus properly tokenized with spaces separating tokens, newlines separating documents
    with open("CleanedCorpus.txt", "w", encoding="utf-8") as f:
        for doc in all_cleaned_documents:
            f.write(" ".join(doc) + "\n")

if __name__ == "__main__":
    scrape_urls()
