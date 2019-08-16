from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
#from fake_useragent import UserAgent
import random
import logging


""" User agent helper class. Allows to get random user agents.
"""
class UserAgent(object):
    ua = None

    """ Returns a random user agent (if database is available),
        otherwise a default one.
        TODO include default user agent table
    """
    @classmethod
    def random(cls):
        if cls.ua is None:
            logger = logging.getLogger(__name__)
            logger.info('Initializing UserAgent using fake_useragent...')
            try:
                cls.ua = fake_useragent.UserAgent()
                # you may replace this by .load()
                cls.ua.load_cached()
            except:
                logger.error('Error initializing UserAgent using fake_useragent.')
                logger.info('Falling back to default User-Agent.')
                cls.ua = False
                return UserAgent.random()
        elif cls.ua is not False:
            return cls.ua.random()
        else:
            # specify a default user agent
            return 'Mozilla/5.0 (Windows NT 6.1; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.99 Safari/537.36'


# Retrieve a random index proxy (we need the index to delete it if not working)
def random_proxy():
  return random.randint(0, len(proxies) - 1)

ua = UserAgent()
proxies = [] # Will contain proxies [ip, port]


# Main function
def main():
    # Retrieve latest proxies
    proxies_req = Request('https://www.sslproxies.org/')
    proxies_req.add_header('User-Agent', ua.random())
    proxies_doc = urlopen(proxies_req).read().decode('utf8')

    soup = BeautifulSoup(proxies_doc, 'html.parser')
    proxies_table = soup.find(id='proxylisttable')

    # Save proxies in the array
    for row in proxies_table.tbody.find_all('tr'):
    proxies.append({
      'ip':   row.find_all('td')[0].string,
      'port': row.find_all('td')[1].string
    })
    return([proxy['ip'] + ':' + proxy['port'] for proxy in proxies])

  # Choose a random proxy
  proxy_index = random_proxy()
  proxy = proxies[proxy_index]

  for n in range(1, 100):
    req = Request('http://icanhazip.com')
    req.set_proxy(proxy['ip'] + ':' + proxy['port'], 'http')

    # Every 10 requests, generate a new proxy
    if n % 10 == 0:
      proxy_index = random_proxy()
      proxy = proxies[proxy_index]

    # Make the call
    try:
      my_ip = urlopen(req).read().decode('utf8')
      print('#' + str(n) + ': ' + my_ip)
    except: # If error, delete this proxy and find another one
      del proxies[proxy_index]
      print('Proxy ' + proxy['ip'] + ':' + proxy['port'] + ' deleted.')
      proxy_index = random_proxy()
      proxy = proxies[proxy_index]




proxy = random_proxy()

if __name__ == '__main__':
  main()
