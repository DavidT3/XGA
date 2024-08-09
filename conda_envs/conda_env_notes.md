conda export --from-history>environment.yml

That makes the environment file.

As it stands you need to remove the pyabel entry on macs with ARM processers - they don't have a release for pyabel on 
that platform specifically, though you can use pip to install it and it works fine.

Then in theory I would run:

conda-lock -f environment.yml -p osx-64 -p linux-64 -p osx-arm64

to make a multi-platform lock file, though that doesn't work for much the same reason as above.

Have tried making a multi-platform lock file without pyabel and it does work.

Also, when installing from an environment file, it seems a good idea to reset the repo priority - it may say
that its on flexible but sometimes it seems to be lying:

conda config --set channel_priority strict
conda config --set channel_priority flexible
