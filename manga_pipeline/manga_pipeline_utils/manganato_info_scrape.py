from bs4 import BeautifulSoup

async def get_manga_info(session, url):
    async with session.get(url) as response:
        text = await response.text()
        soup = BeautifulSoup(text, 'html.parser')
        try:
            manga_info = []
            author = []
            genre = []
            manga_info.append(url)
            info = soup.find('div', {'class': 'story-info-right'})
            try:
                for tag in info.findAll('a', {'class': 'a-h'}):
                    try:
                        if tag['rel'][0] == 'nofollow':
                            author.append(tag.text.strip())
                    except:
                        genre.append(tag.text.strip())
                manga_info.append(author)
                manga_info.append(len(author))
                manga_info.append(genre)
                manga_info.append(len(genre))
            except:
                pass
            try:
                for tag in info.findAll('td', {'class': 'table-value'}):
                    if tag.text.strip() == 'Completed' or tag.text.strip() == 'completed':
                        manga_info.append('finished')
                        break
                    elif tag.text.strip() == 'Ongoing' or tag.text.strip() == 'ongoing' \
                            or 'Ongoing' in tag.text.strip() or 'ongoing' in tag.text.strip():
                        manga_info.append('ongoing')
                        break
            except:
                pass
            try:
                for tag in info.findAll('p'):
                    try:
                        check_view = tag.find('span', {'class': 'stre-label'})
                        if check_view.text.strip() == 'View :':
                            view = tag.find('span', {'class': 'stre-value'}).text.strip()
                    except:
                        continue
                manga_info.append(view)
            except:
                pass
            try:
                num_ratings = info.find('em', {'property': 'v:votes'})
                avg_rating = info.find('em', {'property': 'v:average'})
                manga_info.append(num_ratings.text.strip())
                manga_info.append(avg_rating.text.strip())
            except:
                manga_info.append(0)
                manga_info.append(0)
                pass
        except:
            pass
        try:
            info = soup.find('div', {'class': 'panel-story-chapter-list'})
            last_chapter = info.find('a', {'class': 'chapter-name text-nowrap'}).text.strip()
            manga_info.append(last_chapter)
        except:
            pass
        try:
            info = soup.find('div', {'class': 'panel-story-info-description'})
            description = info.text.strip()
            manga_info.append(description)
        except:
            pass
        return manga_info

