"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Trash2 } from "lucide-react"

type Course = {
  id: number
  name: string
  credits: number
  grade: string
}

const gradePoints: Record<string, number> = {
  A: 4.0,
  "A-": 3.7,
  "B+": 3.3,
  B: 3.0,
  "B-": 2.7,
  "C+": 2.3,
  C: 2.0,
  "C-": 1.7,
  "D+": 1.3,
  D: 1.0,
  F: 0.0,
}

export function GpaCalculator() {
  const [courses, setCourses] = useState<Course[]>([{ id: 1, name: "Course 1", credits: 3, grade: "A" }])
  const [gpa, setGpa] = useState<number | null>(null)

  const addCourse = () => {
    const newId = courses.length > 0 ? Math.max(...courses.map((c) => c.id)) + 1 : 1
    setCourses([...courses, { id: newId, name: `Course ${newId}`, credits: 3, grade: "A" }])
  }

  const removeCourse = (id: number) => {
    setCourses(courses.filter((course) => course.id !== id))
  }

  const updateCourse = (id: number, field: keyof Course, value: string | number) => {
    setCourses(courses.map((course) => (course.id === id ? { ...course, [field]: value } : course)))
  }

  const calculateGpa = () => {
    if (courses.length === 0) {
      setGpa(0)
      return
    }

    let totalPoints = 0
    let totalCredits = 0

    courses.forEach((course) => {
      const points = gradePoints[course.grade] * course.credits
      totalPoints += points
      totalCredits += course.credits
    })

    setGpa(totalCredits > 0 ? totalPoints / totalCredits : 0)
  }

  return (
    <Card className="retro-border bg-secondary/50">
      <CardHeader className="pb-2">
        <CardTitle className="text-sm terminal-text">GPA Calculator</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 terminal-text text-xs">
        <div className="space-y-2">
          {courses.map((course) => (
            <div key={course.id} className="flex items-center space-x-2">
              <div className="flex-1">
                <Input
                  value={course.name}
                  onChange={(e) => updateCourse(course.id, "name", e.target.value)}
                  className="h-7 text-xs"
                  placeholder="Course name"
                />
              </div>
              <div className="w-16">
                <Input
                  type="number"
                  value={course.credits}
                  onChange={(e) => updateCourse(course.id, "credits", Number.parseInt(e.target.value) || 0)}
                  className="h-7 text-xs"
                  placeholder="Credits"
                  min={1}
                  max={6}
                />
              </div>
              <div className="w-16">
                <Select value={course.grade} onValueChange={(value) => updateCourse(course.id, "grade", value)}>
                  <SelectTrigger className="h-7 text-xs">
                    <SelectValue placeholder="Grade" />
                  </SelectTrigger>
                  <SelectContent>
                    {Object.keys(gradePoints).map((grade) => (
                      <SelectItem key={grade} value={grade} className="text-xs">
                        {grade}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => removeCourse(course.id)}
                className="h-7 w-7"
                disabled={courses.length <= 1}
              >
                <Trash2 className="h-3 w-3" />
              </Button>
            </div>
          ))}
        </div>

        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={addCourse} className="text-xs h-7">
            Add Course
          </Button>
          <Button variant="default" size="sm" onClick={calculateGpa} className="text-xs h-7">
            Calculate GPA
          </Button>
        </div>

        {gpa !== null && (
          <div className="mt-4 text-center">
            <div className="text-lg font-bold retro-glow">GPA: {gpa.toFixed(2)}</div>
          </div>
        )}
      </CardContent>
    </Card>
  )
}
