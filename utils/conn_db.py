import mysql.connector

# Configuration for connecting to the MySQL database
db_config = {
    "host": "mysql", # Change localhost if dont run docker
    "user": "Nhan",
    "password": "Nh@n",
    "database": "tensorbot",
    "port": "3306"
}

# Decorator to handle database connection and cursor management
def connect_sql(func):
    def wrap(*args, **kwargs):
        # Establish a database connection using the provided configuration
        conn = mysql.connector.connect(**db_config)
        try:
            # Create a cursor object with dictionary return type
            cursor = conn.cursor(dictionary=True)
            # Execute the wrapped function with the cursor and additional arguments
            result = func(cursor, *args, **kwargs)
            # Commit the transaction if no exception occurred
            conn.commit()
            return result
        finally:
            # Close the cursor after executing the function
            cursor.close()
            # Closing the connection is commented out to prevent connection issues
            # conn.close()
    return wrap

@connect_sql
def get_list_target(cursor):
    # Query to select all targets except those with the value "Block"
    query = 'SELECT target FROM target_dict WHERE target != "Block"'
    cursor.execute(query)
    # Fetch all results from the executed query
    result = cursor.fetchall()
    # Extract the target values, convert them to lowercase, and join them into a single string separated by "|"
    targets = [row['target'].lower() for row in result]
    list_target = "|".join(targets)
    return list_target

@connect_sql
def retrieval_coordinates(cursor, target_to_retrieve):
    # Query to select coordinates for a specific target
    query = "SELECT coordinate FROM target_dict WHERE target = %s"
    cursor.execute(query, (target_to_retrieve,))
    # Fetch one result from the executed query
    result = cursor.fetchone()
    coordinates_list = None
    if result:
        # Evaluate the string representation of coordinates to convert it into a list
        coordinates_list = eval(result.get('coordinate', '[]'))
        # Uncomment the line below to print the type of the first coordinate (for debugging purposes)
        # print(f"Coordinates for {target_to_retrieve}: {type(coordinates_list[0])}")
    # Uncomment the else block to print a message if no coordinates were found (for debugging purposes)
    # else:
    #     print(f"No coordinates found for {target_to_retrieve}")
    return coordinates_list

@connect_sql
def retrieval_info(cursor, target_to_retrieve):
    # Query to select info for a specific target
    query = "SELECT info FROM target_dict WHERE target = %s"
    cursor.execute(query, (target_to_retrieve,))
    # Fetch one result from the executed query
    result = cursor.fetchone()
    return result["info"]

@connect_sql
def update_target_coordinates(cursor, target, new_coordinates):
    # Query to update coordinates for a specific target
    query = "UPDATE target_dict SET coordinate = %s WHERE target = %s;"
    cursor.execute(query, (str(new_coordinates), target))
    print("Update successful")

@connect_sql
def update_target_info(cursor, target, new_info):
    # Query to update info for a specific target
    query = "UPDATE target_dict SET info = %s WHERE target = %s;"
    cursor.execute(query, (new_info, target))
    print("Update successful")

# Example SQL insert statements to add data to the target_dict table
# INSERT INTO target_dict (target, coordinate) VALUES ('E1305', '[0,79]');
# INSERT INTO target_dict (target, coordinate, info) VALUES ('E1305', '[0,79]', NULL);
# INSERT INTO target_dict (target, info) VALUES ('SCADA', 'SCADA stands for "Supervisory Control and Data Acquisition", and is an automation control and monitoring system widely used in industrial processes and facilities. Infrastructure.');

if __name__ == "__main__":
    # Example usage of update_target_coordinates function
    # print(get_list_target())  # Uncomment to print the list of targets
    update_target_coordinates("E1304", [1, 27])
