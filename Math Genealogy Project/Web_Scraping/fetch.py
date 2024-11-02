import asyncio
import json

import aiohttp
import async_timeout

from parse import parse



ERROR_STRING = 'You have specified an ID that does not exist in the database.'
errors = {}
data = []


# folder to save the data
output_path = "..\\Data\\scraping_results\\"

# name of the data
output_name = "data.json"


'''
Performs web scrapping,
saves the result in a .json file called data, with a single key called "data"
Every entry is a mathematician, a dictionary with format

{
  "students": [
    int, int, ...   <-- refers to the id field
  ],
  "advisors": [
    int, int, ...
  ],
  "name": str,
  "school": str,
  "subject": str,
  "thesis": str,
  "country": str,
  "year": int,
  "id": int,
}

'''


# try to load
print('Loading any existing data')
try:
    with open(output_path + output_name, 'r') as infile:
        data = json.load(infile)['data']
    print('Found existing data')
except Exception as e:
    print('No existing data found')


# open metadata
try:
    with open('metadata.json', 'r') as infile:
        metadata = json.load(infile)
except Exception as e:
    pass



# dont read existing data
existing = set(x['id'] for x in data)
print('Skipping {} known records'.format(len(existing)))


# to execute (concurrent)
sem = asyncio.BoundedSemaphore(5)
loop = asyncio.get_event_loop()


# set params
id_min = metadata['id_min']
id_max = metadata['id_max']
bad_ids = set(metadata.get('bad_ids', []))
max_found = id_max
try_further = max_found + 300



print(f"Read ID from {id_min} to {try_further}")

# helper functions

async def fetch(session, url):
    async with async_timeout.timeout(10):
        async with session.get(url) as response:
            print('fetching {}'.format(url))
            return await response.text()


async def fetch_by_id(session, mgp_id):
    async with sem:
        url = 'https://genealogy.math.ndsu.nodak.edu/id.php?id={}'.format(
            mgp_id)
        raw_html = await fetch(session, url)

        if ERROR_STRING in raw_html:
            print('bad id={}'.format(mgp_id))
            bad_ids.add(mgp_id)
            return

        failed = False
        info_dict = {}

        try:
            info_dict = parse(mgp_id, raw_html)
        except Exception as e:
            print('Failed to parse id={}'.format(mgp_id))
            failed = e
        finally:
            if failed:
                errors[mgp_id] = failed
            else:
                data.append(info_dict)


# main function
async def main():
    async with aiohttp.ClientSession(loop=loop) as session:
        tasks = [asyncio.create_task(fetch_by_id(session, i)) for i in range(id_min, try_further + 1) if i not in existing]
        await asyncio.wait(tasks)


# run main
loop.run_until_complete(main())



# save the results


# save errors
with open('errors.txt', 'w') as outfile:
    for i, error in errors.items():
        outfile.write('{},{}\n'.format(i, error))



# sava data (principal)
with open(output_path + output_name, 'w') as outfile:
    json.dump({'data': data}, outfile)



# update metadata
processed = set(x['id'] for x in data)
with open('metadata.json', 'w') as outfile:
    json.dump(
        {
            'id_min': id_min,
            'id_max': max(processed),
            'bad_ids': list(bad_ids),
        }, outfile)


# save the date of the scrapping

from datetime import datetime

# Get the current date
current_date = datetime.now().strftime("%Y-%m-%d")

# Write the date to a text file
with open(output_path + "scraping_date.txt", "w") as file:
    file.write(f"Date of the web scraping: {current_date}")


print('Done!')
