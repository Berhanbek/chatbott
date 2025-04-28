"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { PageHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Trash2, Plus, Calculator, ArrowRight, Award, TrendingUp, AlertTriangle } from "lucide-react"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { motion, AnimatePresence } from "framer-motion"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { cn } from "@/lib/utils"

type Course = {
  id: number
  name: string
  credits: number
  grade: string
}

type GradePoint = {
  grade: string
  minScore: number
  maxScore: number
  points: number
}

const gradePoints: GradePoint[] = [
  { grade: "A+/A", minScore: 85, maxScore: 100, points: 4.0 },
  { grade: "A-", minScore: 80, maxScore: 84.99, points: 3.75 },
  { grade: "B+", minScore: 75, maxScore: 79.99, points: 3.5 },
  { grade: "B", minScore: 68, maxScore: 74.99, points: 3.0 },
  { grade: "B-", minScore: 65, maxScore: 67.99, points: 2.75 },
  { grade: "C+", minScore: 60, maxScore: 64.99, points: 2.5 },
  { grade: "C", minScore: 50, maxScore: 59.99, points: 2.0 },
  { grade: "C-", minScore: 45, maxScore: 49.99, points: 1.75 },
  { grade: "D", minScore: 40, maxScore: 44.99, points: 1.0 },
  { grade: "F", minScore: 0, maxScore: 39.99, points: 0.0 },
]

const studentYears = ["First Year", "Second Year", "Third Year", "Fourth Year"]

export default function GpaCalculator() {
  const [courses, setCourses] = useState<Course[]>([{ id: 1, name: "Course 1", credits: 3, grade: "A+/A" }])
  const [gpa, setGpa] = useState<number | null>(null)
  const [currentGpa, setCurrentGpa] = useState<string>("")
  const [targetGpa, setTargetGpa] = useState<string>("")
  const [studentYear, setStudentYear] = useState<string>("First Year")
  const [requiredGpa, setRequiredGpa] = useState<number | null>(null)
  const [showSuccess, setShowSuccess] = useState(false)
  const [sliderValue, setSliderValue] = useState<number[]>([3.0])
  const [isCalculating, setIsCalculating] = useState(false)

  // Update target GPA when slider changes
  useEffect(() => {
    setTargetGpa(sliderValue[0].toFixed(2))
  }, [sliderValue])

  const addCourse = () => {
    const newId = courses.length > 0 ? Math.max(...courses.map((c) => c.id)) + 1 : 1
    setCourses([...courses, { id: newId, name: `Course ${newId}`, credits: 3, grade: "A+/A" }])
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
    let totalCreditHours = 0

    courses.forEach((course) => {
      const gradePoint = gradePoints.find((gp) => gp.grade === course.grade)
      if (gradePoint) {
        const points = gradePoint.points * course.credits
        totalPoints += points
        totalCreditHours += course.credits
      }
    })

    setGpa(totalCreditHours > 0 ? totalPoints / totalCreditHours : 0)
    setShowSuccess(true)

    // Hide success message after 3 seconds
    setTimeout(() => {
      setShowSuccess(false)
    }, 3000)
  }

  const calculateRequiredGpa = () => {
    setIsCalculating(true)

    // Simulate calculation delay for UI feedback
    setTimeout(() => {
      const current = Number.parseFloat(currentGpa)
      const target = Number.parseFloat(targetGpa)

      if (isNaN(current) || isNaN(target)) {
        setRequiredGpa(null)
        setIsCalculating(false)
        return
      }

      // Get remaining years based on current year
      let remainingYears = 4
      switch (studentYear) {
        case "First Year":
          remainingYears = 4
          break
        case "Second Year":
          remainingYears = 3
          break
        case "Third Year":
          remainingYears = 2
          break
        case "Fourth Year":
          remainingYears = 1
          break
      }

      // Estimate credits per year (30 per semester, 60 per year)
      const completedYears = 4 - remainingYears
      const estimatedCompletedCredits = completedYears * 60
      const estimatedRemainingCredits = remainingYears * 60

      // Calculate required GPA for remaining credits
      const requiredGpaValue =
        (target * (estimatedCompletedCredits + estimatedRemainingCredits) - current * estimatedCompletedCredits) /
        estimatedRemainingCredits
      setRequiredGpa(requiredGpaValue)
      setIsCalculating(false)
    }, 1000)
  }

  const resetCalculator = () => {
    setCourses([{ id: 1, name: "Course 1", credits: 3, grade: "A+/A" }])
    setGpa(null)
  }

  const getGradeColor = (gpa: number) => {
    if (gpa >= 3.5) return "text-green-500"
    if (gpa >= 3.0) return "text-blue-500"
    if (gpa >= 2.0) return "text-yellow-500"
    return "text-red-500"
  }

  const getDifficultyText = (gpa: number) => {
    if (gpa > 4) return "Impossible to achieve"
    if (gpa > 3.7) return "Very challenging"
    if (gpa > 3.3) return "Challenging"
    if (gpa > 3.0) return "Moderate"
    if (gpa > 2.5) return "Achievable"
    if (gpa > 0) return "Easily achievable"
    return "Already achieved!"
  }

  const getDifficultyColor = (gpa: number) => {
    if (gpa > 4) return "text-red-500"
    if (gpa > 3.7) return "text-orange-500"
    if (gpa > 3.3) return "text-yellow-500"
    if (gpa > 3.0) return "text-blue-500"
    if (gpa > 0) return "text-green-500"
    return "text-green-500"
  }

  return (
    <div className="container py-8 px-4 md:px-6">
      <PageHeader title="GPA Calculator" description="Calculate your semester GPA or plan your academic goals" />

      <Tabs defaultValue="calculator" className="mt-8">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="calculator">Semester GPA</TabsTrigger>
          <TabsTrigger value="predictor">GPA Predictor</TabsTrigger>
        </TabsList>

        <TabsContent value="calculator" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>Semester GPA Calculator</CardTitle>
              <CardDescription>
                Enter your courses, credit hours, and grades to calculate your GPA for the semester
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="grid grid-cols-12 gap-4 font-medium text-sm">
                  <div className="col-span-5 md:col-span-6">Course Name</div>
                  <div className="col-span-3 md:col-span-2">Credits</div>
                  <div className="col-span-3 md:col-span-3">Grade</div>
                  <div className="col-span-1"></div>
                </div>

                <AnimatePresence>
                  {courses.map((course) => (
                    <motion.div
                      key={course.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.2 }}
                      className="grid grid-cols-12 gap-4 items-center"
                    >
                      <div className="col-span-5 md:col-span-6">
                        <Input
                          value={course.name}
                          onChange={(e) => updateCourse(course.id, "name", e.target.value)}
                          placeholder="Course name"
                        />
                      </div>
                      <div className="col-span-3 md:col-span-2">
                        <Input
                          type="number"
                          value={course.credits}
                          onChange={(e) => updateCourse(course.id, "credits", Number(e.target.value) || 0)}
                          placeholder="Credits"
                          min={1}
                          max={6}
                        />
                      </div>
                      <div className="col-span-3 md:col-span-3">
                        <Select value={course.grade} onValueChange={(value) => updateCourse(course.id, "grade", value)}>
                          <SelectTrigger>
                            <SelectValue placeholder="Grade" />
                          </SelectTrigger>
                          <SelectContent>
                            {gradePoints.map((gradePoint) => (
                              <SelectItem key={gradePoint.grade} value={gradePoint.grade} className="text-xs">
                                {gradePoint.grade} ({gradePoint.points})
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="col-span-1 flex justify-end">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => removeCourse(course.id)}
                          disabled={courses.length <= 1}
                          className="hover:bg-red-500/10 hover:text-red-500 transition-colors"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>

                <div className="flex flex-col sm:flex-row gap-4 pt-4">
                  <Button variant="outline" onClick={addCourse} className="flex-1">
                    <Plus className="h-4 w-4 mr-2" />
                    Add Course
                  </Button>
                  <Button variant="default" onClick={calculateGpa} className="flex-1 bg-primary">
                    <Calculator className="h-4 w-4 mr-2" />
                    Calculate GPA
                  </Button>
                  <Button variant="secondary" onClick={resetCalculator} className="flex-1">
                    Reset
                  </Button>
                </div>

                <AnimatePresence>
                  {gpa !== null && (
                    <motion.div
                      initial={{ opacity: 0, scale: 0.9 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={{ opacity: 0, scale: 0.9 }}
                      className="mt-6 p-6 bg-muted rounded-lg text-center"
                    >
                      <h3 className="text-lg font-medium mb-2">Your Semester GPA</h3>
                      <div className="relative">
                        <motion.p
                          className={`text-5xl font-bold ${getGradeColor(gpa)}`}
                          initial={{ scale: 1 }}
                          animate={showSuccess ? { scale: [1, 1.1, 1] } : {}}
                          transition={{ duration: 0.5 }}
                        >
                          {gpa.toFixed(2)}
                        </motion.p>
                        {showSuccess && (
                          <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute top-0 right-0 left-0 flex justify-center"
                          >
                            <Award className="h-8 w-8 text-yellow-500 animate-pulse" />
                          </motion.div>
                        )}
                      </div>
                      <div className="mt-4">
                        <Progress value={Math.min(gpa * 25, 100)} className="h-2" />
                      </div>
                      <p className="mt-2 text-sm text-muted-foreground">
                        {gpa >= 3.5
                          ? "Excellent work!"
                          : gpa >= 3.0
                            ? "Great job!"
                            : gpa >= 2.0
                              ? "Good effort!"
                              : "Keep working hard!"}
                      </p>
                    </motion.div>
                  )}
                </AnimatePresence>

                <div className="mt-4 text-xs text-muted-foreground">
                  <p>Note: This calculator provides an estimate. Refer to your official transcript for final GPA.</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="mt-8">
            <CardHeader>
              <CardTitle>AAU Grading Scale</CardTitle>
              <CardDescription>Addis Ababa University grading system</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b">
                      <th className="text-left py-2 px-4">Letter Grade</th>
                      <th className="text-left py-2 px-4">Score Range</th>
                      <th className="text-left py-2 px-4">Grade Points</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gradePoints.map((gradePoint, index) => (
                      <tr key={index} className="border-b hover:bg-muted/50 transition-colors">
                        <td className="py-2 px-4">{gradePoint.grade}</td>
                        <td className="py-2 px-4">
                          {gradePoint.minScore} - {gradePoint.maxScore}
                        </td>
                        <td className="py-2 px-4">{gradePoint.points.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="predictor" className="mt-6">
          <Card>
            <CardHeader>
              <CardTitle>GPA Predictor</CardTitle>
              <CardDescription>
                Calculate the GPA you need in your remaining years to achieve your target GPA
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="space-y-6">
                  <div className="space-y-2">
                    <label htmlFor="studentYear" className="text-sm font-medium">
                      Current Year
                    </label>
                    <Select value={studentYear} onValueChange={setStudentYear}>
                      <SelectTrigger id="studentYear">
                        <SelectValue placeholder="Select your current year" />
                      </SelectTrigger>
                      <SelectContent>
                        {studentYears.map((year) => (
                          <SelectItem key={year} value={year}>
                            {year}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label htmlFor="currentGpa" className="text-sm font-medium">
                      Current GPA
                    </label>
                    <Input
                      id="currentGpa"
                      type="number"
                      placeholder="e.g., 3.2"
                      value={currentGpa}
                      onChange={(e) => setCurrentGpa(e.target.value)}
                      min="0"
                      max="4"
                      step="0.01"
                      className="transition-all focus:ring-2 focus:ring-primary"
                    />
                  </div>

                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <label htmlFor="targetGpa" className="text-sm font-medium">
                        Target GPA: {targetGpa}
                      </label>
                      <span
                        className={cn(
                          "text-xs font-medium",
                          Number(targetGpa) >= 3.5
                            ? "text-green-500"
                            : Number(targetGpa) >= 3.0
                              ? "text-blue-500"
                              : Number(targetGpa) >= 2.0
                                ? "text-yellow-500"
                                : "text-red-500",
                        )}
                      >
                        {Number(targetGpa) >= 3.5
                          ? "Excellent"
                          : Number(targetGpa) >= 3.0
                            ? "Very Good"
                            : Number(targetGpa) >= 2.0
                              ? "Good"
                              : "Needs Improvement"}
                      </span>
                    </div>
                    <Slider
                      defaultValue={[3.0]}
                      max={4.0}
                      min={0}
                      step={0.01}
                      value={sliderValue}
                      onValueChange={setSliderValue}
                      className="py-4"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground">
                      <span>0.0</span>
                      <span>1.0</span>
                      <span>2.0</span>
                      <span>3.0</span>
                      <span>4.0</span>
                    </div>
                  </div>

                  <Button
                    onClick={calculateRequiredGpa}
                    className="w-full bg-primary relative overflow-hidden group"
                    disabled={isCalculating}
                  >
                    {isCalculating ? (
                      <span className="flex items-center">
                        <svg
                          className="animate-spin -ml-1 mr-3 h-4 w-4 text-white"
                          xmlns="http://www.w3.org/2000/svg"
                          fill="none"
                          viewBox="0 0 24 24"
                        >
                          <circle
                            className="opacity-25"
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="currentColor"
                            strokeWidth="4"
                          ></circle>
                          <path
                            className="opacity-75"
                            fill="currentColor"
                            d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                          ></path>
                        </svg>
                        Calculating...
                      </span>
                    ) : (
                      <>
                        <span className="relative z-10 flex items-center">
                          <ArrowRight className="h-4 w-4 mr-2" />
                          Calculate Required GPA
                        </span>
                        <span className="absolute inset-0 h-full w-0 bg-gradient-to-r from-primary to-secondary/80 transition-all duration-300 group-hover:w-full"></span>
                      </>
                    )}
                  </Button>
                </div>

                <div className="flex items-center justify-center">
                  <AnimatePresence mode="wait">
                    {isCalculating ? (
                      <motion.div
                        key="calculating"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-center"
                      >
                        <div className="w-16 h-16 border-4 border-primary border-t-transparent rounded-full animate-spin mx-auto"></div>
                        <p className="mt-4 text-muted-foreground">Calculating your path to success...</p>
                      </motion.div>
                    ) : requiredGpa !== null ? (
                      <motion.div
                        key="result"
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        exit={{ opacity: 0, scale: 0.9 }}
                        className="text-center"
                      >
                        <h3 className="text-lg font-medium mb-4">Required GPA for Remaining Years</h3>
                        <div className="relative">
                          <motion.div
                            className="p-6 bg-muted rounded-full h-40 w-40 flex items-center justify-center mx-auto"
                            initial={{ boxShadow: "0 0 0 rgba(16, 163, 127, 0)" }}
                            animate={{ boxShadow: "0 0 20px rgba(16, 163, 127, 0.3)" }}
                            transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, repeatType: "reverse" }}
                          >
                            <p
                              className={`text-4xl font-bold ${requiredGpa < 0 || requiredGpa > 4 ? "text-red-500" : getGradeColor(requiredGpa)}`}
                            >
                              {requiredGpa < 0 || requiredGpa > 4 ? "N/A" : requiredGpa.toFixed(2)}
                            </p>
                          </motion.div>
                          {requiredGpa > 0 && requiredGpa <= 4 && (
                            <motion.div
                              className="absolute -top-2 -right-2"
                              initial={{ rotate: 0 }}
                              animate={{ rotate: 10, scale: [1, 1.1, 1] }}
                              transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, repeatType: "reverse" }}
                            >
                              <TrendingUp className="h-8 w-8 text-primary" />
                            </motion.div>
                          )}
                          {(requiredGpa < 0 || requiredGpa > 4) && (
                            <motion.div
                              className="absolute -top-2 -right-2"
                              initial={{ scale: 1 }}
                              animate={{ scale: [1, 1.1, 1] }}
                              transition={{ duration: 0.5, repeat: Number.POSITIVE_INFINITY }}
                            >
                              <AlertTriangle className="h-8 w-8 text-yellow-500" />
                            </motion.div>
                          )}
                        </div>
                        <div className="mt-4 space-y-2">
                          <p className={`font-medium ${getDifficultyColor(requiredGpa)}`}>
                            {getDifficultyText(requiredGpa)}
                          </p>
                          <p className="text-sm text-muted-foreground">
                            {requiredGpa < 0
                              ? "Your target GPA is already achieved!"
                              : requiredGpa > 4
                                ? "Your target GPA is not achievable with the remaining time."
                                : `You need to maintain a ${requiredGpa.toFixed(2)} GPA in your remaining years.`}
                          </p>
                          {requiredGpa > 0 && requiredGpa <= 4 && (
                            <div className="pt-2">
                              <p className="text-xs text-muted-foreground mb-1">Difficulty level:</p>
                              <div className="h-2 w-full bg-muted-foreground/20 rounded-full overflow-hidden">
                                <motion.div
                                  className={cn(
                                    "h-full",
                                    requiredGpa > 3.7
                                      ? "bg-red-500"
                                      : requiredGpa > 3.3
                                        ? "bg-orange-500"
                                        : requiredGpa > 3.0
                                          ? "bg-yellow-500"
                                          : requiredGpa > 2.5
                                            ? "bg-blue-500"
                                            : "bg-green-500",
                                  )}
                                  initial={{ width: "0%" }}
                                  animate={{ width: `${Math.min((requiredGpa / 4) * 100, 100)}%` }}
                                  transition={{ duration: 1, delay: 0.2 }}
                                ></motion.div>
                              </div>
                            </div>
                          )}
                        </div>
                      </motion.div>
                    ) : (
                      <motion.div
                        key="empty"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="text-center p-6"
                      >
                        <div className="bg-muted p-8 rounded-lg">
                          <Award className="h-12 w-12 text-muted-foreground/40 mx-auto mb-4" />
                          <p className="text-muted-foreground">
                            Enter your current GPA, year, and target GPA to calculate the GPA you need to achieve in
                            your remaining years.
                          </p>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  )
}
