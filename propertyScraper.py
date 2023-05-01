#!/usr/bin/python
# -*- coding: utf-8 -*-

import urllib.request
import urllib.parse
import urllib.error
import bs4 as bs
import ssl
import json
import ast
import os
from urllib.request import Request, urlopen

# For ignoring SSL certificate errors

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

# Input from user

url = input('Enter Zillow House Listing Url- ')

# Making the website believe that you are accessing it using a mozilla browser

req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
webpage = urlopen(req).read()

# Creating a BeautifulSoup object of the html page for easy extraction of data.

soup = BeautifulSoup(webpage, 'html.parser')
print(soup)
html = soup.prettify('utf-8')
property_json = {}
property_json['Details_Broad'] = {}
property_json['Address'] = {}

# Extract Title of the property listing

for title in soup.findAll('title'):
    property_json['Title'] = title.text.strip()
    break

for meta in soup.findAll('meta', attrs={'name': 'description'}):
    property_json['Detail_Short'] = meta['content'].strip()

for div in soup.findAll('div', attrs={'class': 'character-count-truncated'}):
    property_json['Details_Broad']['Description'] = div.text.strip()
    print(property_json['Details_Broad']['Description'])

for (i, script) in enumerate(soup.findAll('script',
                             attrs={'type': 'application/ld+json'})):
    if i == 0:
        json_data = json.loads(script.text)
        print(json_data)
        property_json['Details_Broad']['Number of Rooms'] = json_data['numberOfRooms']
        property_json['Details_Broad']['Floor Size (in sqft)'] = json_data['floorSize']['value']
        property_json['Address']['Street'] = json_data['address']['streetAddress']
        property_json['Address']['Locality'] = json_data['address']['addressLocality']
        property_json['Address']['Region'] = json_data['address']['addressRegion']
        property_json['Address']['Postal Code'] = json_data['address']['postalCode']
    if i == 1:
        json_data = json.loads(script.text)
        property_json['Price in $'] = json_data['offers']['price']
        property_json['Image'] = json_data['image']
        break

with open('data.json', 'w') as outfile:
    json.dump(property_json, outfile, indent=4)

with open('output_file.html', 'wb') as file:
    file.write(html)

print ('----------Extraction of data is complete. Check json file.----------')