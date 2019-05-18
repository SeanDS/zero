"""Datasheet fetcher"""

import logging
import json

from .parts import Part, nonesorter
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
            keyword = f"*{keyword}*"

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
