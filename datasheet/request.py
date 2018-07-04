import os
import sys
import subprocess
import requests
import json
import logging
import tempfile
import progressbar

LOGGER = logging.getLogger("datasheet")

class DatasheetRequest:
    """Datasheet request handler"""
    API_ENDPOINT = "https://octopart.com/api/v3/parts/match"
    API_KEY = "ebdc07fc"
    def __init__(self, keyword, exact=False):
        self.keyword = keyword
        self.exact = exact
        self.parts = None

        self._request()
    
    def _request(self):
        """Request datasheet"""
        # build parameters
        params = {"include[]": "datasheets",
                  "queries": self.search_query}

        # add defaults
        params = {**params, **self.default_params}
        # get parts
        self._handle_response(requests.get(self.API_ENDPOINT, params))

    def _handle_response(self, response):
        """Handle response"""
        if response.status_code != 200:
            raise Exception(response)
        
        if "application/json" not in response.headers["content-type"]:
            raise Exception("unknown response content type")
    
        response_data = response.json()

        # debug info
        LOGGER.debug("request took %d ms", response_data["msec"])

        if not "results" in response_data:
            raise Exception("unexpected response")
        
        # first list item in results
        results = response_data["results"][0]

        # parts
        parts = results["items"]

        LOGGER.debug("%d %s found", len(parts), ["part", "parts"][len(parts) != 1])

        # store parts
        self._parse_parts(parts)

    def _parse_parts(self, raw_parts):
        """Parse parts"""
        parts = []
        for part in raw_parts:
            parts.append(Part(part))
        
        self.parts = parts

    @property
    def search_query(self):
        """Search query JSON string"""
        keyword = self.keyword
        if not self.exact:
            keyword = "*%s*" % keyword
        
        return json.dumps([{"mpn": keyword}])

    @property
    def default_params(self):
        """Default parameters to include in every request"""
        return {"apikey": self.API_KEY}
    
    @property
    def n_parts(self):
        """Number of parts found"""
        return len(self.parts)

class Downloadable:
    def __init__(self, info_stream=sys.stdout):
        self.info_stream = info_stream

    def download(self, url, progress=True):
        if progress:
            stream = self.info_stream
        else:
            # null file
            stream = open(os.devnull, "w")

        pbar = progressbar.ProgressBar(widgets=['Downloading: ',
                                                progressbar.Percentage(),
                                                progressbar.Bar(),
                                                progressbar.ETA()],
                                       max_value=100, fd=stream).start()
        
        # make request
        request = requests.get(url, stream=True)
        total_data_length = int(request.headers.get("content-length"))

        data_length = 0

        tmp = tempfile.NamedTemporaryFile(delete=False)

        with open(tmp.name, "wb") as f:
            for chunk in request.iter_content(chunk_size=128):
                if chunk:
                    f.write(chunk)
                
                    data_length += len(chunk)

                    if data_length == total_data_length:
                        fraction = 100
                    else:
                        fraction = 100 * data_length / total_data_length
                
                    pbar.update(fraction)
        
        return tmp.name

class Part:
    def __init__(self, part_info):
        self.brand = None
        self.brand_url = None
        self.manufacturer = None
        self.manufacturer_url = None
        self.mpn = None
        self.url = None
        self.datasheets = None

        self._parse(part_info)
    
    def _parse(self, part_info):
        self.brand = part_info["brand"]["name"]
        self.brand_url = part_info["brand"]["homepage_url"]
        self.manufacturer = part_info["manufacturer"]["name"]
        self.manufacturer_url = part_info["manufacturer"]["homepage_url"]
        self.mpn = part_info["mpn"]
        self.url = part_info["octopart_url"]
        self.datasheets = [Datasheet(datasheet) for datasheet in part_info["datasheets"]]

    @property
    def n_datasheets(self):
        return len(self.datasheets)

    def __repr__(self):
        return "{brand} / {manufacturer} {mpn}".format(**self.__dict__)

class Datasheet(Downloadable):
    def __init__(self, datasheet_data):
        self.created = None
        self.n_pages = None
        self.url = None

        self._parse(datasheet_data)

        super().__init__()
    
    def _parse(self, datasheet_data):
        self.created = datasheet_data["metadata"]["date_created"]
        self.n_pages = int(datasheet_data["metadata"]["num_pages"])
        self.url = datasheet_data["url"]

    def display(self):
        self._open_pdf(self.download(url=self.url))

    def _open_pdf(self, filename):
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.call([opener, filename])