import { Card, CardContent } from "@/components/ui/card"
import { BookOpen, Users, Calculator, School } from "lucide-react"

export function FeaturesSection() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      <Card className="card-hover">
        <CardContent className="p-6 flex flex-col items-center text-center">
          <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
            <BookOpen className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Department Info</h3>
          <p className="text-sm text-muted-foreground">
            Learn about the School of Information Science, its mission, and facilities
          </p>
        </CardContent>
      </Card>

      <Card className="card-hover">
        <CardContent className="p-6 flex flex-col items-center text-center">
          <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
            <School className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Program Details</h3>
          <p className="text-sm text-muted-foreground">
            Explore the Bachelor of Science in Information Systems curriculum
          </p>
        </CardContent>
      </Card>

      <Card className="card-hover">
        <CardContent className="p-6 flex flex-col items-center text-center">
          <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
            <Users className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Faculty & Staff</h3>
          <p className="text-sm text-muted-foreground">
            Meet the dedicated instructors and staff of the SIS department
          </p>
        </CardContent>
      </Card>

      <Card className="card-hover">
        <CardContent className="p-6 flex flex-col items-center text-center">
          <div className="h-12 w-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
            <Calculator className="h-6 w-6 text-primary" />
          </div>
          <h3 className="text-lg font-semibold mb-2">GPA Calculator</h3>
          <p className="text-sm text-muted-foreground">Calculate your semester GPA and plan your academic goals</p>
        </CardContent>
      </Card>
    </div>
  )
}
