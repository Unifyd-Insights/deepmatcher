import os
import shutil
import zipfile
import tarfile
import gzip
from ..utils import download_from_url

def download(name, urls, root, check=None):
    """Download and unzip an online archive (.zip, .gz, or .tgz).

    Arguments:
        root (str): Folder to download data to.
        check (str or None): Folder whose existence indicates
            that the dataset has already been downloaded, or
            None to check the existence of root/{cls.name}.

    Returns:
        str: Path to extracted dataset.
    """
    path = os.path.join(root, name)
    check = path if check is None else check
    if not os.path.isdir(check):
        for url in urls:
            if isinstance(url, tuple):
                url, filename = url
            else:
                filename = os.path.basename(url)
            zpath = os.path.join(path, filename)
            if not os.path.isfile(zpath):
                if not os.path.exists(os.path.dirname(zpath)):
                    os.makedirs(os.path.dirname(zpath))
                print('downloading {}'.format(filename))
                download_from_url(url, zpath)
            zroot, ext = os.path.splitext(zpath)
            _, ext_inner = os.path.splitext(zroot)
            if ext == '.zip':
                with zipfile.ZipFile(zpath, 'r') as zfile:
                    print('extracting')
                    zfile.extractall(path)
            # tarfile cannot handle bare .gz files
            elif ext == '.tgz' or ext == '.gz' and ext_inner == '.tar':
                with tarfile.open(zpath, 'r:gz') as tar:
                    dirs = [member for member in tar.getmembers()]
                    tar.extractall(path=path, members=dirs)
            elif ext == '.gz':
                with gzip.open(zpath, 'rb') as gz:
                    with open(zroot, 'wb') as uncompressed:
                        shutil.copyfileobj(gz, uncompressed)

      return os.path.join(path, cls.dirname)