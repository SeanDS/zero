"""Datasheet fetcher"""

import os
import sys
import subprocess
import re
import datetime
import json
import logging
import dateutil.parser

from ..misc import Downloadable
from ..config import ZeroConfig

LOGGER = logging.getLogger(__name__)
CONF = ZeroConfig()


class PartRequest(Downloadable, list):
    """Part request handler"""
    def __init__(self, keyword, partial=True, path=None, timeout=None, **kwargs):
        super().__init__(**kwargs)

        self.keyword = keyword
        self.partial = partial
        self.path = path
        self.timeout = timeout

        self._request()

    def _request(self):
        """Request datasheet"""
        # build parameters
        params = {"include[]": "datasheets",
                  "queries": self.search_query}

        # add defaults
        params = {**params, **self.default_params}
        # get parts
        self._handle_response(*self.fetch(CONF["octopart"]["api_endpoint"], params=params,
                                          label="Downloading part information"))

    def _handle_response(self, data, response):
        """Handle response"""
        if response.status_code != 200:
            raise Exception(response)

        if "application/json" not in response.headers["content-type"]:
            raise Exception("unknown response content type")

        response_data = json.loads(data)

        # debug info
        LOGGER.debug("request took %d ms", response_data["msec"])

        if not "results" in response_data:
            raise Exception("unexpected response")

        # first list item in results
        results = next(iter(response_data["results"]))

        # parts
        parts = results["items"]

        LOGGER.debug("%d %s found", len(parts), ["part", "parts"][len(parts) != 1])

        # store parts
        self._parse_parts(parts)

    def _parse_parts(self, raw_parts):
        """Parse parts"""
        for part in raw_parts:
            self.append(Part(part, path=self.path, timeout=self.timeout, progress=self.progress))

    @property
    def search_query(self):
        """Search query JSON string"""
        keyword = self.keyword
        if self.partial:
            keyword = "*%s*" % keyword

        return json.dumps([{"mpn": keyword}])

    @property
    def default_params(self):
        """Default parameters to include in every request"""
        return {"apikey": CONF["octopart"]["api_key"]}

    @property
    def latest_part(self):
        # sort by latest datasheet
        parts = sorted(self, reverse=True, key=lambda part: nonesorter(part.latest_datasheet))

        return next(iter(parts), None)


class Part:
    def __init__(self, part_info, path=None, timeout=None, progress=False):
        self.path = path
        self.timeout = timeout
        self.progress = progress

        self.brand = None
        self.brand_url = None
        self.manufacturer = None
        self.manufacturer_url = None
        self.mpn = None
        self.url = None
        self.datasheets = []

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
            self.datasheets = [Datasheet(datasheet, part_name=self.mpn, path=self.path,
                                         timeout=self.timeout, progress=self.progress)
                               for datasheet in part_info["datasheets"]]

    @property
    def n_datasheets(self):
        return len(self.datasheets)

    @property
    def sorted_datasheets(self):
        # order datasheets
        return sorted(self.datasheets, reverse=True, key=nonesorter)

    @property
    def latest_datasheet(self):
        return next(iter(self.sorted_datasheets), None)

    def __repr__(self):
        return "{brand} / {manufacturer} {mpn}".format(**self.__dict__)


class Datasheet(Downloadable):
    def __init__(self, datasheet_data, part_name=None, path=None, **kwargs):
        """Datasheet.

        Parameters
        ----------
        path : :class:`str` or file
            Path to store downloaded file. If a directory is specified, the downloaded part name is
            used.
        """
        super().__init__(**kwargs)

        self.part_name = part_name
        self.path = path
        self.created = None
        self.n_pages = None
        self.url = None

        # flag for whether datasheet PDF has been downloaded
        self._downloaded = False

        self._parse(datasheet_data)

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

        if os.path.isdir(path):
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

        filename, _ = self.fetch_file(url=self.url, filename=self.path,
                                      label="Downloading %s" % self.part_name)

        # update path, if necessary
        self.path = filename

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


def nonesorter(datasheet):
    """Return datasheet creation date, or minimum time, for the purposes of sorting."""
    if getattr(datasheet, "created", None) is None:
        # use minimum date
        zero = datetime.datetime.min
        zero.replace(tzinfo=None)
        return zero

    return datasheet.created.replace(tzinfo=None)
