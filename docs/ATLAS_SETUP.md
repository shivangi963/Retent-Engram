# Using MongoDB Atlas (cloud) instead of the local MongoDB

The app reads one setting, `MONGO_URI`, so moving to Atlas needs no code change.

1. Create a free account at https://www.mongodb.com/atlas and create a free **M0** cluster (512 MB is plenty).
2. **Database Access** -> add a database user (username + password). Avoid `@ : / ?` in the password,
   or URL-encode them.
3. **Network Access** -> add your current IP address (for a quick demo you can allow `0.0.0.0/0`;
   remove it afterwards).
4. **Connect -> Drivers** -> copy the connection string. It looks like
   `mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/` - put your real password in place of `PASSWORD`.
5. Install the SRV helper: `pip install dnspython`
6. Copy the data you already have (safe to repeat, never duplicates):

   ```
   python scripts/mongo_copy.py --from mongodb://localhost:27017 --to "mongodb+srv://USER:PASSWORD@cluster0.xxxxx.mongodb.net/"
   ```
7. In `.env` set `MONGO_URI=` to the Atlas string and restart the app (and the API if you use it).
8. Check: `python scripts/selfcheck.py` should print `MongoDB (local or Atlas) - connected - Atlas (cloud)`.

Keep `.env` out of Git (it is in `.gitignore`). To go back to local MongoDB, set
`MONGO_URI=mongodb://localhost:27017`.
