import requests
import json

class DatasheetRequest:
    """Datasheet request handler"""
    API_ENDPOINT = "https://octopart.com/api/v3/parts/match"
    API_KEY = "ebdc07fc"
    def __init__(self, keyword):
        self.keyword = keyword
    
    def request(self):
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
        
        if not response.headers["content-type"].contains("application/json"):
            raise Exception("unknown response content type")
    
        print(response)

    @property
    def search_query(self):
        """Search query JSON string"""
        return json.dumps({"mpn": self.keyword})

    @property
    def default_params(self):
        """Default parameters to include in every request"""
        return {"apikey": self.API_KEY}