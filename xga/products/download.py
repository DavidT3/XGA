#  This code is part of X-ray: Generate and Analyse (XGA), a module designed for the XMM Cluster Survey (XCS).
#  Last modified by David J Turner (djturner@umbc.edu) 9/29/26, 10:49 AM. Copyright (c) The Contributors.

import os
from collections.abc import Sequence

from fsspec.callbacks import DEFAULT_CALLBACK, TqdmCallback
from fsspec.core import url_to_fs
from tqdm.auto import tqdm as auto_tqdm

from xga.products import BaseProduct


def download_products(
    products: Sequence[BaseProduct],
    save_path: str,
    remote_file_sys=None,
    overwrite: bool = False,
    batch_size: int | None = None,
    disable_progress: bool = False,
) -> dict:
    """
    Downloads remote files for a batch of BaseProduct instances, reusing fsspec filesystem connections
    per unique (protocol, fsspec_kwargs) combination, and submitting each group's files as a single
    batched fsspec 'get' call - async-capable filesystems (s3fs, gcsfs, HTTPFileSystem) parallelise these
    transfers internally, which is generally the fastest approach for many small remote files (e.g. S3).

    If a group's batched transfer fails outright, falls back to downloading each product in that group
    individually so one bad file doesn't block the rest. Any product that ultimately fails has 'usable'
    set to False and 'ProductDownloadFailed' added to 'not_usable_reasons'.

    :param list products: BaseProduct (or subclass) instances to download.
    :param str save_path: Directory (preferred) or complete file path (single-product case only).
    :param remote_file_sys: An fsspec filesystem to use for every product. If None, one is built per unique
        protocol/fsspec_kwargs combination found in 'products'.
    :param bool overwrite: Force re-download of already-downloaded products. Default False.
    :param int batch_size: Passed through to fsspec's 'get' if the filesystem supports it, to cap
        concurrent transfers per group. Default None (filesystem default).
    :param bool disable_progress: Setting this to True turns off the download progress bar. Default False.
    :return: Dict mapping each product to its local path, or to the Exception raised on failure.
    :rtype: dict
    """
    # This is what we'll return - a mapping of product instance to either a successful local path, or
    #  an Exception if something went wrong further down. Built up incrementally as we go.
    results = {}

    # First pass over all the products - we're splitting them into those that don't need fetching at all
    #  (already local, in-memory, or already downloaded and we're not forcing an overwrite) and those that
    #  actually need to go and get a remote file. No point setting up filesystem connections for products
    #  we're not going to touch.
    to_fetch = []
    for cur_prod in products:
        if cur_prod.not_from_file or cur_prod.remote_path is None or (cur_prod.downloaded and not overwrite):
            # Nothing to do for this one - if it's local/in-memory then cur_prod.path is already the right answer,
            #  and if it's already downloaded then cur_prod.path was updated to the local copy by a previous call
            results[cur_prod] = cur_prod.path
        else:
            to_fetch.append(cur_prod)

    # If literally everything was already handled in the loop above, we can bail out early
    if not to_fetch:
        return results

    # Now we need to decide how to group the products that need fetching by filesystem, so that we only
    #  construct (and thus authenticate/connect) each distinct filesystem once, and can then batch all the
    #  relevant files through a single 'get' call on that filesystem for maximum efficiency.
    if remote_file_sys is not None:
        # The user has explicitly supplied a filesystem to use for everything, so there's only one group -
        #  we trust that this filesystem is actually compatible with every product's remote location, as
        #  that's the user's responsibility for taking control of this argument
        prod_groups = {"user_supplied": (remote_file_sys, to_fetch)}
    else:
        # No filesystem was supplied, so we build our own groupings. Products are considered to belong to
        #  the same group (and thus can share a filesystem instance) if they have the same remote protocol
        #  (e.g. s3, https) AND the same fsspec_kwargs (e.g. same credentials/anon setting) - two products
        #  with different credentials for the same protocol cannot safely share a filesystem connection
        grouped: dict[tuple, list[BaseProduct]] = {}
        for cur_prod in to_fetch:
            # We turn the fsspec_kwargs dictionary into a sorted tuple of items so that it is hashable and
            #  can be used as (part of) a dictionary key - dictionaries themselves aren't hashable
            key = (cur_prod.remote_type, tuple(sorted((cur_prod.fsspec_kwargs or {}).items())))
            grouped.setdefault(key, []).append(cur_prod)

        # Now we actually construct one filesystem instance per unique key identified above, using the
        #  first product in each group as the representative for building the filesystem (they should all
        #  be equivalent within a group by construction)
        prod_groups = {}
        for key, grp_products in grouped.items():
            try:
                grp_fs, _ = url_to_fs(grp_products[0].remote_path, **(grp_products[0].fsspec_kwargs or {}))
                prod_groups[key] = (grp_fs, grp_products)
            except Exception as err:
                # If we can't even build the filesystem for this group (e.g. bad credentials, unsupported
                #  protocol), then every product in the group is a lost cause - mark them all as unusable
                #  and record the exception, rather than letting this stop the rest of the batch
                for cur_prod in grp_products:
                    cur_prod._usable = False
                    if "ProductDownloadFailed" not in cur_prod.not_usable_reasons:
                        cur_prod.not_usable_reasons.append("ProductDownloadFailed")
                    results[cur_prod] = err

    # Now we actually perform the downloads, one filesystem group at a time
    for grp_fs, grp_products in prod_groups.values():
        # These three lists are built up in parallel (same index = same product) so that after the batched
        #  'get' call succeeds, we know exactly which local path corresponds to which product
        remote_fs_paths = []
        local_paths = []
        mapping = []

        # Working out, once per group, whether the user's save_path should be treated as a directory (the
        #  preferred usage, so that each remote file keeps its own name) or as a single complete file path
        #  (only really sensible if there's one file in this group, but we don't enforce that here)
        cur_save_path = os.fspath(save_path)
        is_dir_like = (
            cur_save_path.endswith(os.sep) or os.path.isdir(cur_save_path) or os.path.splitext(cur_save_path)[1] == ""
        )
        if is_dir_like:
            # Make sure the destination directory actually exists before we try to download anything into it
            os.makedirs(cur_save_path, exist_ok=True)

        # Building the parallel lists of remote paths and local destination paths
        for cur_prod in grp_products:
            _, remote_fs_path = url_to_fs(cur_prod.remote_path, **(cur_prod.fsspec_kwargs or {}))
            local_path = os.path.join(cur_save_path, os.path.basename(remote_fs_path)) if is_dir_like else cur_save_path
            remote_fs_paths.append(remote_fs_path)
            local_paths.append(local_path)
            mapping.append((cur_prod, local_path))

        cb = (
            DEFAULT_CALLBACK
            if disable_progress
            else TqdmCallback(tqdm_kwargs={"desc": "Downloading products"}, tqdm_cls=auto_tqdm)
        )
        try:
            # This is the key efficiency step - passing lists of remote and local paths to a single 'get'
            #  call, rather than looping one product at a time, allows async-capable filesystems (s3fs,
            #  gcsfs, fsspec's HTTPFileSystem) to fetch multiple files concurrently under the hood
            with cb:
                get_kwargs = {} if batch_size is None else {"batch_size": batch_size}
                grp_fs.get(remote_fs_paths, local_paths, callback=cb, **get_kwargs)

            # If we get here without an exception, we assume the whole batch succeeded - update every
            #  product in this group to point at its new local file, and mark it as downloaded
            for cur_prod, local_path in mapping:
                cur_prod._path = local_path
                cur_prod._local_file = True
                cur_prod._downloaded = True
                results[cur_prod] = local_path
        except Exception:
            # The batched transfer failed for the group as a whole - rather than giving up on every product
            #  in the group, we fall back to downloading them one at a time through the normal instance
            #  method, so that a single problematic file doesn't take down the rest of a perfectly good
            #  batch. Any individual failures here are caught and recorded, not raised, so the loop over
            #  prod_groups can continue uninterrupted
            for cur_prod, _ in mapping:
                try:
                    cur_prod.download(save_path, overwrite=overwrite, remote_file_sys=grp_fs, show_warn=False)
                    results[cur_prod] = cur_prod.path
                except Exception as err:
                    # cur_prod.download() already sets cur_prod._usable = False and appends to cur_prod._why_unusable
                    #  internally on failure, so we just need to record the exception here
                    results[cur_prod] = err

    return results
