"use client"

import { useState, useRef } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { PageHeader } from "@/components/page-header"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Search, Mail, MapPin, Users, Phone, ChevronDown, ChevronUp } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"
import { Button } from "@/components/ui/button"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"
import { cn } from "@/lib/utils"

type FacultyMember = {
  id: string
  name: string
  email: string
  office: string
  role?: string
  phone?: string
  specialization?: string
  bio?: string
  image?: string
}

export default function Faculty() {
  const [searchQuery, setSearchQuery] = useState("")
  const [expandedFaculty, setExpandedFaculty] = useState<string | null>(null)
  const [activeFilter, setActiveFilter] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement>(null)

  const departmentHead: FacultyMember = {
    id: "head-1",
    name: "Dr. Michael Melese",
    email: "michael.melese@aau.edu.et",
    office: "Eshetu Chole 621",
    role: "Department Head",
    phone: "+251 911 234 567",
    specialization: "Information Systems Management, Digital Transformation",
    bio: "Dr. Michael Melese has been leading the School of Information Science since 2019. He holds a PhD in Information Systems from the University of Manchester and has published extensively in the field of digital transformation in developing economies.",
    image: "/placeholder.svg?height=400&width=400",
  }

  const coordinators: FacultyMember[] = [
    {
      id: "coord-1",
      name: "Ato Betsegaw Dereje",
      email: "betsegaw.dereje@aau.edu.et",
      office: "Room 423",
      role: "Coordinator",
      phone: "+251 911 123 456",
      specialization: "Software Engineering, Web Technologies",
      bio: "Ato Betsegaw coordinates undergraduate programs and specializes in software engineering.",
      image: "/placeholder.svg?height=400&width=400",
    },
    {
      id: "coord-2",
      name: "Dr. Tibebe Beshah",
      email: "tibebe.beshah@aau.edu.et",
      office: "Room 422",
      role: "Coordinator",
      phone: "+251 911 987 654",
      specialization: "Data Science, Machine Learning",
      bio: "Dr. Tibebe coordinates graduate programs and research initiatives. His research focuses on applied machine learning and data science for development.",
      image: "/placeholder.svg?height=400&width=400",
    },
  ]

  const instructors: FacultyMember[] = [
    {
      id: "1",
      name: "W/ro Adey Edessa",
      email: "adey.edessa@aau.edu.et",
      office: "Room 113",
      specialization: "Database Management",
    },
    {
      id: "2",
      name: "W/t Amina Abdulkadir",
      email: "amina.abdulkadir@aau.edu.et",
      office: "Room 122",
      specialization: "Web Technologies",
    },
    {
      id: "3",
      name: "Ato Aminu Mohammed",
      email: "aminu.mohammed@aau.edu.et",
      office: "Room 319",
      specialization: "Mobile Computing",
    },
    {
      id: "4",
      name: "Ato Andargachew Asfaw",
      email: "andargachew.asfaw@aau.edu.et",
      office: "Room 424",
      specialization: "Networking",
    },
    {
      id: "5",
      name: "W/t Dagmawit Mohammed",
      email: "dagmawit.mohammed@aau.edu.et",
      office: "Room 122",
      specialization: "UI/UX Design",
    },
    {
      id: "6",
      name: "Dr. Dereje Teferi",
      email: "dereje.teferi@aau.edu.et",
      office: "Room 419",
      specialization: "Information Security",
    },
    {
      id: "7",
      name: "Dr. Ermias Abebe",
      email: "ermias.abebe@aau.edu.et",
      office: "Room 115",
      specialization: "Artificial Intelligence",
    },
    {
      id: "8",
      name: "Dr. Getachew H/Mariam",
      email: "getachew.hailemariam@aau.edu.et",
      office: "Room 618",
      specialization: "Cloud Computing",
    },
    {
      id: "9",
      name: "Ato Gebremichael Meshesha",
      email: "gebremichael.meshesha@aau.edu.et",
      office: "Room 122",
      specialization: "Programming Paradigms",
    },
    {
      id: "10",
      name: "Ato Kidus Menfes",
      email: "kidus.menfese@aau.edu.et",
      office: "Room 511",
      specialization: "Software Testing",
    },
    {
      id: "11",
      name: "W/o Lemlem Hagos",
      email: "lemlem.hagos@aau.edu.et",
      office: "Room 116",
      specialization: "Project Management",
    },
    {
      id: "12",
      name: "Dr. Lemma Lessa",
      email: "lemma.lessa@aau.edu.et",
      office: "Room 417",
      specialization: "Enterprise Systems",
    },
    {
      id: "13",
      name: "Dr. Martha Yifiru",
      email: "martha.yifiru@aau.edu.et",
      office: "Room 420",
      specialization: "Human-Computer Interaction",
    },
    {
      id: "14",
      name: "Ato Melaku Girma",
      email: "melaku.girma@aau.edu.et",
      office: "Room 224",
      specialization: "Data Structures & Algorithms",
    },
    {
      id: "15",
      name: "W/o Meseret Hailu",
      email: "meseret.hailu@aau.edu.et",
      office: "Room 113",
      specialization: "Database Administration",
    },
    {
      id: "16",
      name: "Dr. Melekamu Beyene",
      email: "melekamu.beyene@aau.edu.et",
      office: "Room 423",
      specialization: "Big Data Analytics",
    },
    {
      id: "17",
      name: "Ato Miftah Hassen",
      email: "miftah.hassen@aau.edu.et",
      office: "Room 424",
      specialization: "Mobile App Development",
    },
    {
      id: "18",
      name: "W/t Mihiret Tibebe",
      email: "mihret.tibebe@aau.edu.et",
      office: "Room 113",
      specialization: "Web Design",
    },
    {
      id: "19",
      name: "Dr. Million Meshesha",
      email: "million.meshesha@aau.edu.et",
      office: "Room 418",
      specialization: "Information Retrieval",
    },
    {
      id: "20",
      name: "Dr. Rahel Bekele",
      email: "rahel.bekele@aau.edu.et",
      office: "Room 221",
      specialization: "E-Learning Systems",
    },
    {
      id: "21",
      name: "Dr. Solomon Tefera",
      email: "solomon.teferra@aau.edu.et",
      office: "Room 421",
      specialization: "Cybersecurity",
    },
    {
      id: "22",
      name: "Dr. Temtem Assefa",
      email: "temtem.assefa@aau.edu.et",
      office: "Room 622",
      specialization: "Data Mining",
    },
    {
      id: "23",
      name: "Ato Teshome Alemu",
      email: "teshome.alemu@aau.edu.et",
      office: "Room 224",
      specialization: "System Analysis",
    },
    {
      id: "24",
      name: "Ato Tsegaye Berhanu",
      email: "tsegaye.berhanu@aau.edu.et",
      office: "Room 319",
      specialization: "Computer Graphics",
    },
    {
      id: "25",
      name: "Ato Wendwesen Endale",
      email: "wendwesen.endale@aau.edu.et",
      office: "Room 319",
      specialization: "Operating Systems",
    },
    {
      id: "26",
      name: "Dr. Wondwossen Mulugeta",
      email: "wondwossen.mulugeta@aau.edu.et",
      office: "Room 114",
      specialization: "Computer Networks",
    },
    {
      id: "27",
      name: "Dr. Workshet Lamenew",
      email: "workshet.lamenew@aau.edu.et",
      office: "Room 222",
      specialization: "Software Architecture",
    },
  ]

  // Get unique specializations for filtering
  const specializations = Array.from(new Set(instructors.map((i) => i.specialization).filter(Boolean) as string[]))

  const toggleFacultyDetails = (id: string) => {
    setExpandedFaculty(expandedFaculty === id ? null : id)

    // Scroll to the expanded card after a short delay
    if (expandedFaculty !== id) {
      setTimeout(() => {
        const element = document.getElementById(`faculty-${id}`)
        if (element) {
          element.scrollIntoView({ behavior: "smooth", block: "center" })
        }
      }, 100)
    }
  }

  const filteredInstructors = instructors.filter((member) => {
    const matchesSearch =
      member.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.email.toLowerCase().includes(searchQuery.toLowerCase()) ||
      member.office.toLowerCase().includes(searchQuery.toLowerCase()) ||
      (member.specialization && member.specialization.toLowerCase().includes(searchQuery.toLowerCase()))

    const matchesFilter = !activeFilter || member.specialization === activeFilter

    return matchesSearch && matchesFilter
  })

  return (
    <div className="container py-8 px-4 md:px-6">
      <PageHeader
        title="Meet the People Behind SIS"
        description="Our faculty and staff are dedicated to excellence in teaching and research"
      />

      <div className="mb-8 mt-8">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search faculty by name, email, office, or specialization..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
      </div>

      <Tabs defaultValue="all" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="all">All Faculty</TabsTrigger>
          <TabsTrigger value="leadership">Leadership</TabsTrigger>
          <TabsTrigger value="instructors">Instructors</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="mt-6">
          <div className="space-y-6" ref={scrollRef}>
            <h3 className="text-xl font-semibold">Department Leadership</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <motion.div whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 300 }}>
                <Card className="overflow-hidden h-full" id={`faculty-${departmentHead.id}`}>
                  <div className="aspect-square relative">
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent z-10"></div>
                    <Avatar className="h-full w-full rounded-none">
                      <AvatarImage
                        src={departmentHead.image || "/placeholder.svg"}
                        alt={departmentHead.name}
                        className="object-cover"
                      />
                      <AvatarFallback className="text-4xl h-full w-full rounded-none bg-primary/20">
                        {departmentHead.name.charAt(0)}
                      </AvatarFallback>
                    </Avatar>
                    <div className="absolute bottom-4 left-4 z-20">
                      <p className="text-white text-lg font-semibold">{departmentHead.name}</p>
                      <p className="text-white/80 text-sm">{departmentHead.role}</p>
                    </div>
                  </div>
                  <CardContent className="p-6">
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center">
                          <Mail className="h-4 w-4 text-primary mr-2" />
                          <a href={`mailto:${departmentHead.email}`} className="text-sm hover:underline">
                            {departmentHead.email}
                          </a>
                        </div>
                        <div className="flex items-center">
                          <Phone className="h-4 w-4 text-primary mr-2" />
                          <p className="text-sm">{departmentHead.phone}</p>
                        </div>
                        <div className="flex items-center">
                          <MapPin className="h-4 w-4 text-primary mr-2" />
                          <p className="text-sm">{departmentHead.office}</p>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Specialization</p>
                        <p className="text-sm text-muted-foreground">{departmentHead.specialization}</p>
                      </div>
                      <div>
                        <p className="text-sm">{departmentHead.bio}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>

              {coordinators.map((coordinator) => (
                <motion.div key={coordinator.id} whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 300 }}>
                  <Card className="overflow-hidden h-full" id={`faculty-${coordinator.id}`}>
                    <div className="aspect-square relative">
                      <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent z-10"></div>
                      <Avatar className="h-full w-full rounded-none">
                        <AvatarImage
                          src={coordinator.image || "/placeholder.svg"}
                          alt={coordinator.name}
                          className="object-cover"
                        />
                        <AvatarFallback className="text-4xl h-full w-full rounded-none bg-primary/20">
                          {coordinator.name.charAt(0)}
                        </AvatarFallback>
                      </Avatar>
                      <div className="absolute bottom-4 left-4 z-20">
                        <p className="text-white text-lg font-semibold">{coordinator.name}</p>
                        <p className="text-white/80 text-sm">{coordinator.role}</p>
                      </div>
                    </div>
                    <CardContent className="p-6">
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <div className="flex items-center">
                            <Mail className="h-4 w-4 text-primary mr-2" />
                            <a href={`mailto:${coordinator.email}`} className="text-sm hover:underline">
                              {coordinator.email}
                            </a>
                          </div>
                          <div className="flex items-center">
                            <Phone className="h-4 w-4 text-primary mr-2" />
                            <p className="text-sm">{coordinator.phone}</p>
                          </div>
                          <div className="flex items-center">
                            <MapPin className="h-4 w-4 text-primary mr-2" />
                            <p className="text-sm">{coordinator.office}</p>
                          </div>
                        </div>
                        <div>
                          <p className="text-sm font-medium">Specialization</p>
                          <p className="text-sm text-muted-foreground">{coordinator.specialization}</p>
                        </div>
                        <div>
                          <p className="text-sm">{coordinator.bio}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}
            </div>

            <div className="section-divider"></div>

            <h3 className="text-xl font-semibold">Instructors</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredInstructors.map((instructor) => (
                <motion.div
                  key={instructor.id}
                  whileHover={{ y: -3 }}
                  transition={{ type: "spring", stiffness: 400 }}
                  id={`faculty-${instructor.id}`}
                >
                  <Card
                    className={cn(
                      "overflow-hidden transition-all duration-300",
                      expandedFaculty === instructor.id ? "ring-2 ring-primary" : "",
                    )}
                  >
                    <CardContent className="p-4">
                      <div className="flex items-start justify-between">
                        <div className="flex items-center">
                          <Avatar className="h-10 w-10 mr-3">
                            <AvatarFallback className="bg-primary/20">{instructor.name.charAt(0)}</AvatarFallback>
                          </Avatar>
                          <div>
                            <h4 className="font-semibold">{instructor.name}</h4>
                            {instructor.specialization && (
                              <p className="text-xs text-primary">{instructor.specialization}</p>
                            )}
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 rounded-full hover:bg-primary/10"
                          onClick={() => toggleFacultyDetails(instructor.id)}
                        >
                          {expandedFaculty === instructor.id ? (
                            <ChevronUp className="h-4 w-4" />
                          ) : (
                            <ChevronDown className="h-4 w-4" />
                          )}
                        </Button>
                      </div>

                      <AnimatePresence>
                        {expandedFaculty === instructor.id && (
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.3 }}
                            className="overflow-hidden"
                          >
                            <div className="pt-4 mt-4 border-t space-y-2">
                              <div className="flex items-center">
                                <Mail className="h-4 w-4 text-muted-foreground mr-2" />
                                <a href={`mailto:${instructor.email}`} className="text-sm hover:underline">
                                  {instructor.email}
                                </a>
                              </div>
                              <div className="flex items-center">
                                <MapPin className="h-4 w-4 text-muted-foreground mr-2" />
                                <p className="text-sm">{instructor.office}</p>
                              </div>
                            </div>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </CardContent>
                  </Card>
                </motion.div>
              ))}

              {filteredInstructors.length === 0 && (
                <div className="col-span-full text-center py-12">
                  <Users className="h-12 w-12 text-muted-foreground/40 mx-auto mb-4" />
                  <p className="text-muted-foreground">No faculty members found matching your search.</p>
                  <Button
                    variant="outline"
                    className="mt-4"
                    onClick={() => {
                      setSearchQuery("")
                      setActiveFilter(null)
                    }}
                  >
                    Clear Filters
                  </Button>
                </div>
              )}
            </div>
          </div>
        </TabsContent>

        <TabsContent value="leadership" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <motion.div whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 300 }}>
              <Card className="overflow-hidden h-full">
                <div className="aspect-square relative">
                  <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent z-10"></div>
                  <Avatar className="h-full w-full rounded-none">
                    <AvatarImage
                      src={departmentHead.image || "/placeholder.svg"}
                      alt={departmentHead.name}
                      className="object-cover"
                    />
                    <AvatarFallback className="text-4xl h-full w-full rounded-none bg-primary/20">
                      {departmentHead.name.charAt(0)}
                    </AvatarFallback>
                  </Avatar>
                  <div className="absolute bottom-4 left-4 z-20">
                    <p className="text-white text-lg font-semibold">{departmentHead.name}</p>
                    <p className="text-white/80 text-sm">{departmentHead.role}</p>
                  </div>
                </div>
                <CardContent className="p-6">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <div className="flex items-center">
                        <Mail className="h-4 w-4 text-primary mr-2" />
                        <a href={`mailto:${departmentHead.email}`} className="text-sm hover:underline">
                          {departmentHead.email}
                        </a>
                      </div>
                      <div className="flex items-center">
                        <Phone className="h-4 w-4 text-primary mr-2" />
                        <p className="text-sm">{departmentHead.phone}</p>
                      </div>
                      <div className="flex items-center">
                        <MapPin className="h-4 w-4 text-primary mr-2" />
                        <p className="text-sm">{departmentHead.office}</p>
                      </div>
                    </div>
                    <div>
                      <p className="text-sm font-medium">Specialization</p>
                      <p className="text-sm text-muted-foreground">{departmentHead.specialization}</p>
                    </div>
                    <div>
                      <p className="text-sm">{departmentHead.bio}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </motion.div>

            {coordinators.map((coordinator) => (
              <motion.div key={coordinator.id} whileHover={{ y: -5 }} transition={{ type: "spring", stiffness: 300 }}>
                <Card className="overflow-hidden h-full">
                  <div className="aspect-square relative">
                    <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent z-10"></div>
                    <Avatar className="h-full w-full rounded-none">
                      <AvatarImage
                        src={coordinator.image || "/placeholder.svg"}
                        alt={coordinator.name}
                        className="object-cover"
                      />
                      <AvatarFallback className="text-4xl h-full w-full rounded-none bg-primary/20">
                        {coordinator.name.charAt(0)}
                      </AvatarFallback>
                    </Avatar>
                    <div className="absolute bottom-4 left-4 z-20">
                      <p className="text-white text-lg font-semibold">{coordinator.name}</p>
                      <p className="text-white/80 text-sm">{coordinator.role}</p>
                    </div>
                  </div>
                  <CardContent className="p-6">
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex items-center">
                          <Mail className="h-4 w-4 text-primary mr-2" />
                          <a href={`mailto:${coordinator.email}`} className="text-sm hover:underline">
                            {coordinator.email}
                          </a>
                        </div>
                        <div className="flex items-center">
                          <Phone className="h-4 w-4 text-primary mr-2" />
                          <p className="text-sm">{coordinator.phone}</p>
                        </div>
                        <div className="flex items-center">
                          <MapPin className="h-4 w-4 text-primary mr-2" />
                          <p className="text-sm">{coordinator.office}</p>
                        </div>
                      </div>
                      <div>
                        <p className="text-sm font-medium">Specialization</p>
                        <p className="text-sm text-muted-foreground">{coordinator.specialization}</p>
                      </div>
                      <div>
                        <p className="text-sm">{coordinator.bio}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>

        <TabsContent value="instructors" className="mt-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredInstructors.map((instructor) => (
              <motion.div key={instructor.id} whileHover={{ y: -3 }} transition={{ type: "spring", stiffness: 400 }}>
                <Card
                  className={cn(
                    "overflow-hidden transition-all duration-300",
                    expandedFaculty === instructor.id ? "ring-2 ring-primary" : "",
                  )}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex items-center">
                        <Avatar className="h-10 w-10 mr-3">
                          <AvatarFallback className="bg-primary/20">{instructor.name.charAt(0)}</AvatarFallback>
                        </Avatar>
                        <div>
                          <h4 className="font-semibold">{instructor.name}</h4>
                          {instructor.specialization && (
                            <p className="text-xs text-primary">{instructor.specialization}</p>
                          )}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 rounded-full hover:bg-primary/10"
                        onClick={() => toggleFacultyDetails(instructor.id)}
                      >
                        {expandedFaculty === instructor.id ? (
                          <ChevronUp className="h-4 w-4" />
                        ) : (
                          <ChevronDown className="h-4 w-4" />
                        )}
                      </Button>
                    </div>

                    <AnimatePresence>
                      {expandedFaculty === instructor.id && (
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: "auto", opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.3 }}
                          className="overflow-hidden"
                        >
                          <div className="pt-4 mt-4 border-t space-y-2">
                            <div className="flex items-center">
                              <Mail className="h-4 w-4 text-muted-foreground mr-2" />
                              <a href={`mailto:${instructor.email}`} className="text-sm hover:underline">
                                {instructor.email}
                              </a>
                            </div>
                            <div className="flex items-center">
                              <MapPin className="h-4 w-4 text-muted-foreground mr-2" />
                              <p className="text-sm">{instructor.office}</p>
                            </div>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </CardContent>
                </Card>
              </motion.div>
            ))}

            {filteredInstructors.length === 0 && (
              <div className="col-span-full text-center py-12">
                <Users className="h-12 w-12 text-muted-foreground/40 mx-auto mb-4" />
                <p className="text-muted-foreground">No faculty members found matching your search.</p>
                <Button
                  variant="outline"
                  className="mt-4"
                  onClick={() => {
                    setSearchQuery("")
                    setActiveFilter(null)
                  }}
                >
                  Clear Filters
                </Button>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}
