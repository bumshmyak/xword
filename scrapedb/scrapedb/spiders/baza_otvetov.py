import scrapy

class BazaSpider(scrapy.Spider):
  name = "baza"

  start_urls = ['https://baza-otvetov.ru/categories/view/24/',]

  def parse(self, response):
    table = response.css('table.q-list__table')[0]
    rows = table.xpath('//tr')
    for row in rows[1:]:
      cols = row.xpath('td//text()')
      yield {
        'question': cols[2].extract(),
        'answer': cols[4].extract(), 
      }

    nav = response.css('div.q-list__nav')
    if not nav:
      return
    nav = nav[0]

    next_page = nav.css('a::attr(href)')[-2].get() 
    if next_page is not None:
      yield response.follow(next_page, callback=self.parse)
