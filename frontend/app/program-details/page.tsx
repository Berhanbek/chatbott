import { Card, CardContent } from "@/components/ui/card"
import { PageHeader } from "@/components/page-header"
import { Button } from "@/components/ui/button"
import { Code, Layout, Shield, BarChartIcon as ChartBar, BookOpen, Users, Sparkles } from "lucide-react"

export default function ProgramDetails() {
  return (
    <div className="container py-8 px-4 md:px-6">
      <PageHeader
        title="Bachelor of Science in Information Systems"
        description="4-year program blending computing, management, and systems thinking"
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-8">
        <Card className="lg:col-span-3">
          <CardContent className="p-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Program Overview</h3>
                <p className="text-sm">
                  A 4-year program preparing students for careers in software development, database design, systems
                  analysis, and IT consulting.
                </p>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Key Focus Areas</h3>
                <ul className="text-sm space-y-1 list-disc pl-4">
                  <li>Technical programming skills</li>
                  <li>Database management</li>
                  <li>Systems analysis and design</li>
                  <li>Project management</li>
                </ul>
              </div>
              <div>
                <h3 className="text-lg font-semibold mb-3 text-primary">Career Paths</h3>
                <ul className="text-sm space-y-1 list-disc pl-4">
                  <li>Software Developer</li>
                  <li>Database Administrator</li>
                  <li>Systems Analyst</li>
                  <li>IT Consultant</li>
                  <li>Business Analyst</li>
                </ul>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="lg:col-span-2 space-y-6">
          <Card>
            <CardContent className="p-6">
              <h3 className="text-xl font-semibold mb-4">Program Structure</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary">Year 1</h4>
                  <p className="text-xs text-muted-foreground">
                    Foundation in computing, mathematics, and general education
                  </p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary">Year 2</h4>
                  <p className="text-xs text-muted-foreground">Programming, database fundamentals, and system design</p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary">Year 3</h4>
                  <p className="text-xs text-muted-foreground">
                    Advanced topics, specialized electives, and project work
                  </p>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary">Year 4</h4>
                  <p className="text-xs text-muted-foreground">Capstone projects and professional practice</p>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="overflow-hidden border-primary/20">
            <div className="bg-gradient-to-r from-primary/20 to-secondary/20 p-4">
              <div className="flex items-center">
                <div className="h-10 w-10 rounded-full bg-background/20 backdrop-blur-sm flex items-center justify-center mr-3">
                  <Sparkles className="h-5 w-5 text-white" />
                </div>
                <div>
                  <h3 className="text-lg font-semibold text-white">Information Science Hub</h3>
                  <p className="text-white/80 text-xs">Student & Department-Led Initiative</p>
                </div>
              </div>
            </div>
            <CardContent className="p-4">
              <div className="grid grid-cols-2 gap-3 mb-4">
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary text-sm flex items-center">
                    <Users className="h-3 w-3 mr-2" />
                    Coding Clubs
                  </h4>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary text-sm flex items-center">
                    <ChartBar className="h-3 w-3 mr-2" />
                    Hackathons
                  </h4>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary text-sm flex items-center">
                    <Layout className="h-3 w-3 mr-2" />
                    Workshops
                  </h4>
                </div>
                <div className="p-3 bg-muted rounded-lg">
                  <h4 className="font-medium text-primary text-sm flex items-center">
                    <Shield className="h-3 w-3 mr-2" />
                    Mentorship
                  </h4>
                </div>
              </div>
              <div className="flex justify-center">
                <a href="https://t.me/InformationSystemsHub" target="_blank" rel="noopener noreferrer">
                  <Button className="bg-primary hover:bg-primary/90 text-sm">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-4 w-4 mr-2"
                      viewBox="0 0 24 24"
                      fill="currentColor"
                    >
                      <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.562 8.248l-1.515 7.144c-.115.54-.42.675-.85.422l-2.35-1.732-1.135 1.092c-.126.12-.232.222-.476.222l.17-2.4 4.37-3.946c.19-.168-.042-.262-.294-.094l-5.406 3.4-2.325-.723c-.505-.156-.514-.505.11-.748l9.083-3.5c.424-.156.794.096.618.863z" />
                    </svg>
                    Join @InformationSystemsHub
                  </Button>
                </a>
              </div>
            </CardContent>
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="bg-primary/5 border-primary/20">
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold mb-3">Core Courses</h3>
              <ul className="space-y-2 text-sm">
                <li className="flex items-center">
                  <BookOpen className="h-4 w-4 text-primary mr-2 flex-shrink-0" />
                  <span>Programming (C++, Java, Web)</span>
                </li>
                <li className="flex items-center">
                  <BookOpen className="h-4 w-4 text-primary mr-2 flex-shrink-0" />
                  <span>Database Systems</span>
                </li>
                <li className="flex items-center">
                  <BookOpen className="h-4 w-4 text-primary mr-2 flex-shrink-0" />
                  <span>System Analysis & Design</span>
                </li>
                <li className="flex items-center">
                  <BookOpen className="h-4 w-4 text-primary mr-2 flex-shrink-0" />
                  <span>Cybersecurity</span>
                </li>
                <li className="flex items-center">
                  <BookOpen className="h-4 w-4 text-primary mr-2 flex-shrink-0" />
                  <span>Data Structures & Algorithms</span>
                </li>
              </ul>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <h3 className="text-lg font-semibold mb-3">Program Highlights</h3>
              <div className="space-y-3">
                <div className="flex items-start">
                  <div className="h-7 w-7 rounded-full bg-secondary/20 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                    <Code className="h-3 w-3 text-secondary" />
                  </div>
                  <div>
                    <p className="font-medium text-sm">Hands-on Learning</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="h-7 w-7 rounded-full bg-secondary/20 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                    <Shield className="h-3 w-3 text-secondary" />
                  </div>
                  <div>
                    <p className="font-medium text-sm">Industry Partnerships</p>
                  </div>
                </div>
                <div className="flex items-start">
                  <div className="h-7 w-7 rounded-full bg-secondary/20 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                    <ChartBar className="h-3 w-3 text-secondary" />
                  </div>
                  <div>
                    <p className="font-medium text-sm">Research Opportunities</p>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
