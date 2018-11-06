"""Datasheet request handler"""

import os
import sys
import subprocess
import re
import datetime
import json
import logging
import tempfile
import requests
import progressbar
import dateutil.parser

from ..config import ZeroConfig

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()


class DatasheetRequest:
    """Datasheet request handler"""
    def __init__(self, keyword, exact=False, path=None):
        self.keyword = keyword
        self.exact = exact
        self.path = path
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
        self._handle_response(requests.get(CONF["octopart"]["api_endpoint"], params))

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
            parts.append(Part(part, path=self.path))

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
        return {"apikey": CONF["octopart"]["api_key"]}

    @property
    def n_parts(self):
        """Number of parts found"""
        return len(self.parts)

    @property
    def latest_datasheet(self):
        # latest datasheet for each part
        datasheets = sorted(self.parts, reverse=True,
                            key=lambda part: part.latest_datasheet.created)

        return next(iter(datasheets), None)


class Downloadable:
    def __init__(self, info_stream=sys.stdout):
        self.info_stream = info_stream

    def fetch(self, url, progress=True):
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

        # create temporary file
        tmp = tempfile.NamedTemporaryFile(delete=False)

        with open(tmp.name, "wb") as file_handler:
            for chunk in request.iter_content(chunk_size=128):
                if chunk:
                    file_handler.write(chunk)

                    data_length += len(chunk)

                    if data_length == total_data_length:
                        fraction = 100
                    else:
                        fraction = 100 * data_length / total_data_length

                    # check in case lengths are misreported
                    if fraction > 100:
                        fraction = 100
                    elif fraction < 0:
                        fraction = 0

                    pbar.update(fraction)

        pbar.finish()

        return tmp.name


class Part:
    def __init__(self, part_info, path=None):
        self.path = path
        self.brand = None
        self.brand_url = None
        self.manufacturer = None
        self.manufacturer_url = None
        self.mpn = None
        self.url = None
        self.datasheets = None

        self._parse(part_info)

    def _parse(self, part_info):
        if part_info.get("brand"):
            if part_info["brand"].get("name"):
                self.brand = part_info["brand"]["name"]
            if part_info["brand"].get("homepage_url"):
                self.brand_url = part_info["brand"]["homepage_url"]
        if part_info.get("manufacturer"):
            if part_info["manufacturer"].get("name"):
                self.manufacturer = part_info["manufacturer"]["name"]
            if part_info["manufacturer"].get("homepage_url"):
                self.manufacturer_url = part_info["manufacturer"]["homepage_url"]
        if part_info.get("mpn"):
            self.mpn = part_info["mpn"]
        if part_info.get("octopart_url"):
            self.url = part_info["octopart_url"]
        if part_info.get("datasheets"):
            self.datasheets = [Datasheet(datasheet, part_name=self.mpn, path=self.path)
                               for datasheet in part_info["datasheets"]]

    @property
    def n_datasheets(self):
        return len(self.datasheets)

    @property
    def sorted_datasheets(self):
        def nonesorter(datasheet):
            if datasheet.created is None:
                # use minimum date
                zero = datetime.datetime.min
                zero.replace(tzinfo=None)
                return zero

            return datasheet.created.replace(tzinfo=None)

        # order datasheets
        return sorted(self.datasheets, reverse=True, key=nonesorter)

    @property
    def latest_datasheet(self):
        return next(iter(self.sorted_datasheets), None)

    def __repr__(self):
        return "{brand} / {manufacturer} {mpn}".format(**self.__dict__)


class Datasheet(Downloadable):
    def __init__(self, datasheet_data, part_name=None, path=None):
        self.part_name = part_name
        self.path = path
        self.created = None
        self.n_pages = None
        self.url = None

        # flag for whether datasheet PDF has been downloaded
        self._downloaded = False

        self._parse(datasheet_data)

        super().__init__()

    def _parse(self, datasheet_data):
        if datasheet_data.get("metadata"):
            if datasheet_data["metadata"].get("date_created"):
                self.created = dateutil.parser.parse(datasheet_data["metadata"]["date_created"])
            if datasheet_data["metadata"].get("num_pages"):
                self.n_pages = int(datasheet_data["metadata"]["num_pages"])
        self.url = datasheet_data["url"]

    @property
    def full_path(self):
        """Get path to store datasheet including filename, or None if no path is set"""
        path = self.path

        if path is not None:
            # add filename
            path = os.path.join(path, self.safe_filename)

        return path

    @property
    def safe_part_name(self):
        """Sanitise part name, generating one if one doesn't exist"""
        part_name = self.part_name

        if self.part_name is None:
            part_name = "unknown"

        return part_name

    @property
    def safe_filename(self):
        """Sanitise filename for storing on the file system"""
        filename = self.safe_part_name
        filename = str(filename).strip().replace(' ', '_')
        filename = re.sub(r'(?u)[^-\w.]', '', filename)

        # add extension
        filename = filename + os.path.extsep + "pdf"

        return filename

    def download(self, force=False):
        if self._downloaded and not force:
            # already downloaded
            return

        tmp_path = self.fetch(url=self.url)

        if self.full_path is not None:
            # move to specified path
            os.rename(tmp_path, self.full_path)

            self.path = os.path.normpath(self.full_path)
        else:
            self.path = tmp_path

        self._downloaded = True

    def display(self):
        self.download()
        self._open_pdf(self.path)

    def _open_pdf(self, filename):
        if sys.platform == "win32":
            os.startfile(filename)
        else:
            opener = "open" if sys.platform == "darwin" else "xdg-open"
            subprocess.run([opener, filename])

    def __str__(self):
        if self.created is not None:
            created = self.created.strftime("%Y-%m-%d")
        else:
            created = "?"

        if self.n_pages is not None:
            pages = self.n_pages
        else:
            pages = "unknown"

        return "Datasheet (created %s, %s pages)" % (created, pages)
