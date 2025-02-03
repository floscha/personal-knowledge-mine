from typing import Iterable, List

from langchain_community.document_loaders.blob_loaders import FileSystemBlobLoader
from langchain_community.document_loaders.blob_loaders.schema import Blob, BlobLoader


class PodcastAudioLoader(BlobLoader):
    """Load podcast audio urls as audio file(s)."""

    def __init__(self, urls: List[str], save_dir: str):
        if not isinstance(urls, list):
            raise TypeError("urls must be a list")

        self.urls = urls
        self.save_dir = save_dir

    def yield_blobs(self) -> Iterable[Blob]:
        """Yield audio blobs for each url."""

        try:
            import lightcast
        except ImportError:
            raise ImportError(
                "lightcast package not found, please install it with "
                "`pip install lightcast`"
            )

        # TODO: Use actual title instead of i
        for i, url in enumerate(self.urls):
            lightcast.download_episode(url, f"{self.save_dir}/{i}.mp3")

        # Yield the written blobs
        loader = FileSystemBlobLoader(self.save_dir, glob="*.mp3")
        for blob in loader.yield_blobs():
            yield blob
