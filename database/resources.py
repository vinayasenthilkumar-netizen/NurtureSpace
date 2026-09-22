
# imports json so resource topics can be converted between stored text and python lists
import json
# imports date and time helpers used for daily resource rotation and bookmark dates
from datetime import date, datetime, timezone
# imports timezone support so singapore dates are handled correctly
from zoneinfo import ZoneInfo
# imports the helper used to open and manage sqlite connections
from database.connection import open_database
# stores the singapore timezone used for daily resources and bookmark date checks
SINGAPORE_TIMEZONE = ZoneInfo("Asia/Singapore")


# converts a sqlite resource row into a dictionary used by the application
def row_to_resource(row):
    #convrt a SQLite resoue row into a dictionary.

    # return nothing when no matching resource row was found
    if row is None:
        return None
    # try to convert the stored topics json back into a python list
    try:
        topics = json.loads(row["topics_json"] or "[]")

    # use an empty list when the stored json cannot be decoded
    except (json.JSONDecodeError, TypeError):
        topics = []
    # make sure topics is actually a list before using it
    if not isinstance(topics, list):
        topics = []
    # build the resource dictionary returned to the rest of the application
    resource = {
        # resource identifier
        "id": row["id"],
        # displayed resource title
        "title": row["title"],
        # article, blog, video or other stored resource type
        "resource_type": row["resource_type"] or "",
        # organisation or source that published the resource
        "source": row["source"] or "",
        # list of topics linked to the resource
        "topics": topics,
        # stored resource description or content
        "content": row["content"] or "",
        # external link for the resource
        "url": row["url"] or "",
        # convert the sqlite active value into a python boolean
        "is_active": bool(row["is_active"]),
        # convert the daily resource flag into a python boolean
        "daily_eligible": bool(row["daily_eligible"]),
    }
    # bookmarked queries include this extra timestamp column
    if "bookmarked_at" in row.keys():
        resource["bookmarked_at"] = row["bookmarked_at"]
    # return the completed resource dictionary
    return resource

# bookmarking 
# converts a stored bookmark timestamp into the correct singapore calendar date
def _bookmark_singapore_date(value):
    
    # convert SQLite bookmark timestamp to Singapore date.
    #sqlite CURRENT_TIMESTAMP is UTC, so this avoids treating late at night Singapore bookmarks as the wrong day.
    # convert the supplied timestamp into clean text
    text = str(value or "").strip()
    # return nothing when there is no timestamp to process
    if not text:
        return None
    # first try to parse an iso formatted timestamp
    try:
        parsed = datetime.fromisoformat(
            # replace a trailing z with an explicit utc offset
            text.replace("Z", "+00:00")
        )

    # try the normal sqlite timestamp format if iso parsing fails
    except ValueError:
        try:
            parsed = datetime.strptime(
                text,
                "%Y-%m-%d %H:%M:%S",
            )
        # return nothing when neither timestamp format can be read
        except ValueError:
            return None
    # sqlite current timestamps are utc when no timezone is included
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    # convert the utc timestamp to singapore time and keep only the date
    return parsed.astimezone(
        SINGAPORE_TIMEZONE
    ).date()

# gets resource ids that were bookmarked before the current singapore date
def get_previous_bookmarked_resource_ids(
    user_id,
    current_date=None,
):
    # Return resources saved before the supplied Singapore date.
    ##Resources saved today are deliberately not excluded from todays Daily Resources.
    # users without an id have no bookmark history to exclude
    if not user_id:
        return set()

    # use todays singapore date when no date is supplied
    if current_date is None:
        current_date = datetime.now(
            SINGAPORE_TIMEZONE
        ).date()

    # retrieve all bookmark ids and their saved timestamps for the user
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT
                resource_id,
                created_at
            FROM resource_bookmarks
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

    # store ids that should be excluded from todays resource rotation
    excluded_ids = set()

    # check each bookmark individually
    for row in rows:
        # convert the stored utc timestamp into a singapore date
        bookmarked_date = _bookmark_singapore_date(
            row["created_at"]
        )

        # exclude only resources bookmarked on an earlier singapore date
        if (
            bookmarked_date is not None
            and bookmarked_date < current_date
        ):
            # store ids as strings so comparisons use one consistent type
            excluded_ids.add(
                str(row["resource_id"])
            )

    # return the set of previously bookmarked resource ids
    return excluded_ids

# Daily resources
# selects the daily curated resources shown on the home page
def get_daily_resources(
    maximum_resources=2,
    current_date=None,
    user_id=None,
):

    #return up to two daily resources.
    #nly active dailyeligible articles, blogs and videos are used.
    #resources saved on an earlier day are excluded

    # use todays singapore date when no testing or custom date is supplied
    if current_date is None:
        current_date = datetime.now(
            SINGAPORE_TIMEZONE
        ).date()

    # retrieve only active resources that are allowed in the daily rotation
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                title,
                resource_type,
                source,
                topics_json,
                content,
                url,
                is_active,
                daily_eligible
            FROM resources
            WHERE is_active = 1
              AND daily_eligible = 1
              AND LOWER(resource_type)
                  IN ('article', 'blog', 'video')
            ORDER BY CAST(id AS INTEGER) ASC
            """
        ).fetchall()

    # convert each sqlite row into the normal resource format
    resources = [
        row_to_resource(row)
        for row in rows
    ]

    # return an empty list when there are no eligible daily resources
    if not resources:
        return []

    # find resources bookmarked before today for this user
    excluded_ids = get_previous_bookmarked_resource_ids(
        user_id=user_id,
        current_date=current_date,
    )

    # keep only resources that have not been bookmarked on an earlier day
    available_resources = [
        resource
        for resource in resources
        if str(resource["id"]) not in excluded_ids
    ]

    # return nothing when every eligible resource has already been excluded
    if not available_resources:
        return []

    # keep the daily result between one and two resources
    maximum_resources = max(
        1,
        min(int(maximum_resources), 2),
    )

    # use the calendar date to choose a repeatable starting point in the list
    start_index = (
        current_date.toordinal()
        % len(available_resources)
    )

    # create the list that will hold todays selected resources
    selected_resources = []

    # select the requested number without exceeding the available resources
    for offset in range(
        min(
            maximum_resources,
            len(available_resources),
        )
    ):
        # move through the list and wrap back to the beginning when needed
        resource_index = (
            start_index + offset
        ) % len(available_resources)

        # add the selected resource to todays list
        selected_resources.append(
            available_resources[resource_index]
        )

    # return the daily resource selection
    return selected_resources

# resoure quieries
# gets every active curated resource stored in the database
def get_active_resources():
    # rreturn all active curated resources.

    # retrieve all active resources in alphabetical title order
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT
                id,
                title,
                resource_type,
                source,
                topics_json,
                content,
                url,
                is_active,
                daily_eligible
            FROM resources
            WHERE is_active = 1
            ORDER BY title ASC
            """
        ).fetchall()

    # convert every database row into the format used by the application
    return [
        row_to_resource(row)
        for row in rows
    ]

# retrieves one curated resource using its id
def get_resource_by_id(resource_id):
    ### return one curated resource by ID

    # convert the supplied id into clean text
    clean_resource_id = str(
        resource_id or ""
    ).strip()

    # return nothing when no usable resource id was provided
    if not clean_resource_id:
        return None

    # retrieve the matching resource from sqlite
    with open_database() as connection:
        row = connection.execute(
            """
            SELECT
                id,
                title,
                resource_type,
                source,
                topics_json,
                content,
                url,
                is_active,
                daily_eligible
            FROM resources
            WHERE id = ?
            """,
            (clean_resource_id,),
        ).fetchone()

    # convert the returned database row before using it elsewhere
    return row_to_resource(row)


# counts the total number of curated resources in the database
def count_resources():
    #reeturn the number of curated resources stored in SQLite

    # ask sqlite to count the stored resource rows
    with open_database() as connection:
        row = connection.execute(
            """
            SELECT COUNT(*) AS resource_count
            FROM resources
            """
        ).fetchone()

    # convert the returned count into a normal integer
    return int(row["resource_count"])


# saves a curated resource to a users bookmarked resources
def bookmark_resource(user_id, resource_id):
    """Save a resource for a user. Duplicate bookmarks are ignored."""

    # a bookmark cannot be saved without a user
    if not user_id:
        raise ValueError(
            "A user ID is required."
        )

    # convert the resource id into clean text
    clean_resource_id = str(
        resource_id or ""
    ).strip()

    # a bookmark cannot be saved without a resource id
    if not clean_resource_id:
        raise ValueError(
            "A resource ID is required."
        )

    # save the bookmark and ignore the request if the same bookmark already exists
    with open_database() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO resource_bookmarks (
                user_id,
                resource_id
            )
            VALUES (?, ?)
            """,
            (
                # user who owns the bookmark
                user_id,

                # resource being bookmarked
                clean_resource_id,
            ),
        )


# removes one bookmarked resource from a users account
def remove_resource_bookmark(
    user_id,
    resource_id,
):
    ## rmove one saved resource from a users account
    # dont do anything when no user id is supplied
    if not user_id:
        return

    # convert the resource id into clean text
    clean_resource_id = str(
        resource_id or ""
    ).strip()

    # do nothing when the resource id is empty
    if not clean_resource_id:
        return

    # delete only the bookmark belonging to this user and resource
    with open_database() as connection:
        connection.execute(
            """
            DELETE FROM resource_bookmarks
            WHERE user_id = ?
              AND resource_id = ?
            """,
            (
                # user who owns the bookmark
                user_id,

                # resource bookmark to remove
                clean_resource_id,
            ),
        )


# checks whether a user has already bookmarked a specific resource
def is_resource_bookmarked(
    user_id,
    resource_id,
):
    # return True when a resource is already bookmarked
    # users without an id cannot have saved bookmarks
    if not user_id:
        return False

    # convert the resource id into clean text
    clean_resource_id = str(
        resource_id or ""
    ).strip()

    # an empty resource id cannot match a bookmark
    if not clean_resource_id:
        return False

    # look for one matching bookmark row
    with open_database() as connection:
        row = connection.execute(
            """
            SELECT 1
            FROM resource_bookmarks
            WHERE user_id = ?
              AND resource_id = ?
            LIMIT 1
            """,
            (
                # user being checked
                user_id,

                # resource being checked
                clean_resource_id,
            ),
        ).fetchone()

    # a returned row means the bookmark already exists
    return row is not None

# gets all resource ids currently bookmarked by a user
def get_bookmarked_resource_ids(user_id):
    #return resource IDs currently bookmarked by a user
    # users without an id have no saved bookmark list
    if not user_id:
        return set()

    # retrieve only the resource ids linked to this user
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT resource_id
            FROM resource_bookmarks
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchall()

    # return unique resource ids as strings
    return {
        str(row["resource_id"])
        for row in rows
    }


# gets the full active resources currently bookmarked by a user
def get_bookmarked_resources(user_id):
    # return active bookmarked resources, newest bookmark first
    # users without an id have no saved resources to return
    if not user_id:
        return []

    # join bookmarks with resources so full resource details can be returned
    with open_database() as connection:
        rows = connection.execute(
            """
            SELECT
                r.id,
                r.title,
                r.resource_type,
                r.source,
                r.topics_json,
                r.content,
                r.url,
                r.is_active,
                r.daily_eligible,
                rb.created_at AS bookmarked_at
            FROM resource_bookmarks rb
            INNER JOIN resources r
                ON r.id = rb.resource_id
            WHERE rb.user_id = ?
              AND r.is_active = 1
            ORDER BY rb.created_at DESC
            """,
            (user_id,),
        ).fetchall()

    # convert all bookmarked database rows into application resource dictionaries
    return [
        row_to_resource(row)
        for row in rows
    ]