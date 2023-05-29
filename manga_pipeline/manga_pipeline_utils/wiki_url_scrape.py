import re
from bs4 import BeautifulSoup
import ast
import aiohttp
import ssl

async def is_wikipedia_page(url, author):
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(url) as response:
                content = await response.text()
                soup = BeautifulSoup(content, 'html.parser')
                page_content = soup.get_text()
                if '200' in str(response):
                    if isinstance(author, str):
                        author = ast.literal_eval(author)
                    for item in author:
                        for word in item.split():
                            author_regex = r'(?i)\b{}\b'.format(re.escape(word))
                            if re.search(author_regex, page_content) and ('serialized' in page_content or 'manga series' in page_content or 'manhwa' in page_content):
                                return True
        return False
    except Exception as e:
        print(f"found exception at {url}: {e}")
        return False

async def find_disambiguation_url(url, author):
    try:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(url) as response:
                content = await response.text()
                soup = BeautifulSoup(content, "html.parser")
                anchor_elements = soup.find_all("a")
                for link_element in anchor_elements:
                    href = link_element.get("href")
                    if href and '(manga)' in href:
                        if await is_wikipedia_page('https://en.wikipedia.org' + href, author):
                            return 'https://en.wikipedia.org' + href
        return 'None'
    except Exception as e:
        print(f"found exception at {url}: {e}")
        return 'None'

async def final_wiki_url_search(title, author):
    try:
        url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&list=search&srsearch={}'.format(title + ' manga')
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=ssl_context)) as session:
            async with session.get(url) as response:
                data = await response.json()
                if 'query' in data and 'search' in data['query']:
                    search_results = data['query']['search']
                    if search_results:
                        first_result = search_results[0]
                        page_title = first_result['title']
                        url = 'https://en.wikipedia.org/w/api.php?action=query&format=json&prop=info&pageids={}&inprop=url'.format(first_result['pageid'])
                        async with session.get(url) as response:
                            data = await response.json()
                            if 'query' in data and 'pages' in data['query']:
                                page_info = data['query']['pages']
                                if page_info:
                                    page = page_info[list(page_info.keys())[0]]
                                    page_url = page['fullurl']
                                    if await is_wikipedia_page(page_url, author):
                                        return page_url
    except Exception as e:
        print(f"found exception at {title}: {e}")
        return 'None'
    return 'None'

async def process_row(row):
    wiki_url = 'https://en.wikipedia.org/wiki/' + row['Title'].strip().split('(')[0].replace(' ', '_') + "_(disambiguation)"
    wiki_url = await find_disambiguation_url(wiki_url, row['author'])
    if wiki_url == 'None':
        wiki_url = await final_wiki_url_search(row['Title'].split('(')[0], row['author'])
    return wiki_url

