from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///employees.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define Employee model
class Employee(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    department = db.Column(db.String(100), nullable=False)

# Create the database tables
with app.app_context():
    db.create_all()

#  GET all employees
@app.route('/employees', methods=['GET'])
def get_employees():
    print("🔹 GET /employees called")  # Debugging
    employees = Employee.query.all()
    
    if not employees:
        return jsonify({"message": "No employees found"}), 404

    return jsonify([{"id": emp.id, "name": emp.name, "department": emp.department} for emp in employees])

#  GET an employee by ID
@app.route('/employees/<int:emp_id>', methods=['GET'])
def get_employee(emp_id):
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({"message": "Employee not found"}), 404
    return jsonify({"id": employee.id, "name": employee.name, "department": employee.department})

#  POST (Add a new employee)
@app.route('/employees', methods=['POST'])
def add_employee():
    data = request.json
    new_employee = Employee(name=data["name"], department=data["department"])
    db.session.add(new_employee)
    db.session.commit()
    return jsonify({"message": "Employee added"}), 201

# PUT (Update an employee)
@app.route('/employees/<int:emp_id>', methods=['PUT'])
def update_employee(emp_id):
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({"message": "Employee not found"}), 404
    
    data = request.json
    employee.name = data.get("name", employee.name)
    employee.department = data.get("department", employee.department)
    db.session.commit()
    return jsonify({"message": "Employee updated"})

# DELETE an employee
@app.route('/employees/<int:emp_id>', methods=['DELETE'])
def delete_employee(emp_id):
    employee = Employee.query.get(emp_id)
    if not employee:
        return jsonify({"message": "Employee not found"}), 404

    db.session.delete(employee)
    db.session.commit()
    return jsonify({"message": "Employee deleted"})

if __name__ == '__main__':
    app.run(debug=True)
